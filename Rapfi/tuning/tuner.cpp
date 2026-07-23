/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "tuner.h"

#include "../config.h"
#include "../core/iohelper.h"
#include "../core/random.h"
#include "../core/time.h"
#include "../eval/eval.h"
#include "../game/board.h"
#include "dataset.h"
#include "optimizer.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <thread>

namespace {

using Tuning::Float;
using Tuning::LossType;
using Tuning::PolicyCandidate;
using Tuning::PreparedCorpus;
using Tuning::TuneCoeff;
using Tuning::TuneGradient;
using Tuning::TuneParam;

constexpr Float CoeffScale = 8;

uint64_t domainSeed(uint64_t seed, uint64_t domain)
{
    PRNG mixer(seed ^ domain);
    return mixer();
}

TuneParam encodeIntegerForTruncatingExport(Score score, Float scale, Float bias)
{
    TuneParam nearest = TuneParam((Float(score) - bias) / scale);
    TuneParam candidates[] = {
        nearest,
        std::nextafter(nearest, -std::numeric_limits<TuneParam>::infinity()),
        std::nextafter(nearest, std::numeric_limits<TuneParam>::infinity()),
    };

    for (TuneParam candidate : candidates) {
        Float reconstructed = Float(candidate) * scale + bias;
        if (std::isfinite(candidate) && std::trunc(reconstructed) == Float(score))
            return candidate;
    }

    throw std::runtime_error(
        "move-score scale and bias cannot represent an integer score without drift");
}

inline bool checkEqual(Float a, Float b)
{
    return std::abs(a - b) <= Float(1.0);
}

/// sigmoid(x) = 1/(1+exp(-x))
inline Float sigmoid(Float x)
{
    return Float(1) / (Float(1) + std::exp(-x));
}

inline Float scoreToWinrate(Float score, Float invScalingFactor)
{
    return sigmoid(score * invScalingFactor);
}

inline Float lossFunction(LossType lt, Float logit, Float target)
{
    Float pred = sigmoid(logit);
    switch (lt) {
    case LossType::L1: return std::abs(target - pred);
    case LossType::L2: {
        Float dist = target - pred;
        return dist * dist;
    }
    case LossType::BCE: {
        Float loss = std::max(logit, Float(0)) - logit * target
                     + std::log1p(std::exp(-std::abs(logit)));
        Float targetBias = 0;
        if (target > 0)
            targetBias += target * std::log(target);
        if (target < 1)
            targetBias += (1 - target) * std::log1p(-target);
        return (loss + targetBias) / 2;
    }
    default: return Float(0);
    }
}

inline Float lossFunctionLogitGrad(LossType lt, Float logit, Float target)
{
    Float pred = sigmoid(logit);
    Float diff = pred - target;
    switch (lt) {
    case LossType::L1:
        return ((Float(0) < diff) - (diff < Float(0))) * pred * (Float(1) - pred);
    case LossType::L2: return Float(2) * diff * pred * (Float(1) - pred);
    case LossType::BCE: return diff / 2;
    default: return Float(0);
    }
}

/// Collect coefficient from eval info
template <typename Collector>
void collectEvalCoeffs(Rule r, const Evaluation::EvalInfo &evalInfo, Collector collect)
{
    Color self = evalInfo.self, oppo = ~self;

    // Collect pattern code coefficient
    for (size_t pcode = 0; pcode < PCODE_NB; pcode++) {
        int coeff[2][SIDE_NB] = {{evalInfo.plyBack[0].pcodeCount[BLACK][pcode],
                                  evalInfo.plyBack[0].pcodeCount[WHITE][pcode]},
                                 {evalInfo.plyBack[1].pcodeCount[BLACK][pcode],
                                  evalInfo.plyBack[1].pcodeCount[WHITE][pcode]}};

        // Coefficient is scaled at 2x
        if (r == RENJU) {
            collect(coeff[0][self] + coeff[1][self], 2, &Evaluation::EVALS[r + self][pcode]);
            collect(-coeff[0][oppo] - coeff[1][oppo], 2, &Evaluation::EVALS[r + oppo][pcode]);
        }
        else {
            collect(coeff[0][self] - coeff[0][oppo] + coeff[1][self] - coeff[1][oppo],
                    2,
                    &Evaluation::EVALS[r][pcode]);
        }
    }

    // Collect threat eval coefficient
    collect(1, 1, &Evaluation::EVALS_THREAT[Evaluation::tableIndex(r, self)][evalInfo.threatMask]);
}

/// Collect coefficient from board pattern codes
template <typename Collector>
void collectMoveScoreCoeffs(Rule r, const Board &board, Collector collect)
{
    Color self = board.sideToMove(), oppo = ~self;

    // Collect scores of all candidate moves
    FOR_EVERY_EMPTY_CAND_POS(&board, pos)
    {
        collect(pos,
                1,
                1,
                &Evaluation::P4SCORES[Evaluation::tableIndex(r, self)][board.pcode<BLACK>(pos)],
                &Evaluation::P4SCORES[Evaluation::tableIndex(r, oppo)][board.pcode<WHITE>(pos)]);
    }
}

template <typename T, typename UnaryOp>
T parallelIndexReduce(BS::thread_pool &pool, size_t count, T init, UnaryOp transform)
{
    if (count == 0)
        return init;

    constexpr size_t LogicalPartitions = 64;
    size_t numBlocks = std::min(count, LogicalPartitions);
    size_t blockSize = (count + numBlocks - 1) / numBlocks;
    std::vector<std::future<T>> futures;
    futures.reserve(numBlocks);

    for (size_t i = 0; i < numBlocks; ++i) {
        size_t blockBegin = i * blockSize;
        size_t blockEnd   = std::min(blockBegin + blockSize, count);
        if (blockBegin == blockEnd)
            break;

        futures.emplace_back(pool.submit_task([blockBegin, blockEnd, transform]() {
            T blockResult = T {};
            for (size_t index = blockBegin; index < blockEnd; ++index)
                blockResult += transform(index);
            return blockResult;
        }));
    }

    T result = init;
    for (auto &future : futures)
        result += future.get();

    return result;
}

Float linearValue(const std::vector<TuneCoeff> &terms,
                  uint32_t                      begin,
                  uint32_t                      end,
                  const std::vector<TuneParam> &params)
{
    Float value = 0;
    for (uint32_t i = begin; i < end; i++)
        value += terms[i].coeff * params[terms[i].index];
    return value;
}

Float policyScore(const PolicyCandidate &candidate, const std::vector<TuneParam> &params)
{
    return params[candidate.indices[0]] + params[candidate.indices[1]];
}

Float computeLinearEval(const PreparedCorpus        &corpus,
                        size_t                       sample,
                        const std::vector<TuneParam> &params)
{
    return linearValue(corpus.evalTerms(),
                       corpus.evalOffsets()[sample],
                       corpus.evalOffsets()[sample + 1],
                       params)
           / CoeffScale;
}

template <bool UseTunedEval>
Float computeEvalLoss(const PreparedCorpus        &corpus,
                      size_t                       sample,
                      const std::vector<TuneParam> &params,
                      Float                        K,
                      LossType                     loss)
{
    if (corpus.evalOffsets()[sample] == corpus.evalOffsets()[sample + 1])
        return 0;

    Float eval = UseTunedEval ? computeLinearEval(corpus, sample, params)
                              : Float(corpus.staticEvals()[sample]);
    Float result  = Float(corpus.results()[sample]) * Float(0.5);
    return lossFunction(loss, eval * K, result);
}

Float computeMoveScoreLoss(const PreparedCorpus        &corpus,
                           size_t                       sample,
                           const std::vector<TuneParam> &params,
                           Float                        gamma)
{
    uint32_t begin = corpus.policyOffsets()[sample];
    uint32_t end   = corpus.policyOffsets()[sample + 1];
    uint16_t best  = corpus.bestCandidates()[sample];
    if (begin == end || best == PreparedCorpus::NoPolicyTarget)
        return 0;

    const auto &candidates = corpus.policyCandidates();
    Float       maxScore   = std::numeric_limits<Float>::lowest();
    for (uint32_t i = begin; i < end; i++)
        maxScore = std::max(maxScore, policyScore(candidates[i], params));

    Float sumExp = 0;
    for (uint32_t i = begin; i < end; i++)
        sumExp += std::exp(policyScore(candidates[i], params) - maxScore);

    Float scoreClass = policyScore(candidates[begin + best], params) - maxScore;
    Float xClass     = std::exp(scoreClass) / sumExp;
    Float focalWeight = std::pow(Float(1) - xClass, gamma);
    return focalWeight * (-scoreClass + std::log(sumExp));
}

void computeEvalGradient(const PreparedCorpus        &corpus,
                         size_t                       sample,
                         std::vector<TuneGradient>   &grads,
                         const std::vector<TuneParam> &params,
                         Float                        K,
                         LossType                     loss)
{
    uint32_t begin = corpus.evalOffsets()[sample];
    uint32_t end   = corpus.evalOffsets()[sample + 1];
    if (begin == end)
        return;

    Float result   = Float(corpus.results()[sample]) * Float(0.5);
    Float logit    = computeLinearEval(corpus, sample, params) * K;
    Float dL_dEval = lossFunctionLogitGrad(loss, logit, result) * K;
    const auto &evalTerms = corpus.evalTerms();
    for (uint32_t i = begin; i < end; i++)
        grads[evalTerms[i].index] += evalTerms[i].coeff * dL_dEval;
}

void computeMoveScoreGradient(const PreparedCorpus        &corpus,
                              size_t                       sample,
                              std::vector<TuneGradient>   &grads,
                              const std::vector<TuneParam> &params,
                              Float                        gamma)
{
    uint32_t begin = corpus.policyOffsets()[sample];
    uint32_t end   = corpus.policyOffsets()[sample + 1];
    uint16_t best  = corpus.bestCandidates()[sample];
    if (begin == end || best == PreparedCorpus::NoPolicyTarget)
        return;

    const auto &candidates = corpus.policyCandidates();
    Float       maxScore   = std::numeric_limits<Float>::lowest();
    for (uint32_t i = begin; i < end; i++)
        maxScore = std::max(maxScore, policyScore(candidates[i], params));

    Float sumExp = 0;
    for (uint32_t i = begin; i < end; i++)
        sumExp += std::exp(policyScore(candidates[i], params) - maxScore);

    Float invSumExp = Float(1) / sumExp;
    Float bestExp = std::exp(policyScore(candidates[begin + best], params) - maxScore);
    Float Pt = bestExp * invSumExp;
    Float logPt = policyScore(candidates[begin + best], params) - maxScore - std::log(sumExp);
    Float PtClamped = std::clamp(Pt, Float(1e-6), Float(1 - 1e-6));
    Float PtLogPtDivPtSub1 = PtClamped / (PtClamped - 1) * logPt;
    Float dFLdCE = std::pow(1 - Pt, gamma) * (gamma * PtLogPtDivPtSub1 + 1);

    for (uint32_t i = begin; i < end; i++) {
        Float probability = std::exp(policyScore(candidates[i], params) - maxScore) * invSumExp;
        Float dCEdScore   = i == begin + best ? Pt - 1 : probability;
        Float dFLdScore   = dFLdCE * dCEdScore;
        grads[candidates[i].indices[0]] += CoeffScale * dFLdScore;
        grads[candidates[i].indices[1]] += CoeffScale * dFLdScore;
    }
}

}  // namespace

namespace Tuning {

Tuner::Tuner(Dataset &trainDataset, Dataset *valDataset, TuningConfig config)
    : config(config)
    , threadPool(config.numThreads != 0
                     ? config.numThreads
                     : std::max<size_t>(std::thread::hardware_concurrency(), 1))
{
    MESSAGEL("Tuner worker threads = " << threadPool.get_thread_count() << ", seed = "
                                       << config.seed << ".");
    MESSAGEL("Start initializing parameters...");
    initParams();

    MESSAGEL("Start initializing tune entries from training dataset...");
    initTuneEntries(trainTuneEntries, trainDataset, true);

    if (valDataset) {
        MESSAGEL("Start initializing tune entries from validation dataset...");
        initTuneEntries(valTuneEntries, *valDataset, false);
    }
}

/// run() runs the tuner for specified epochs. After each epoch completed, callback will be called.
void Tuner::run(size_t epochs, std::function<void(TuningStatistic)> callback)
{
    Time initTime = now();

    // Set Float output precision
    std::cout << std::setprecision(std::min(std::numeric_limits<Float>::digits10, 7)) << std::fixed;

    // Note: the last non-full batch of tune entries will be dropped.
    // Validate this before calibration, which may otherwise scan the corpus
    // many times only to discover that training cannot run.
    size_t numBatches = trainTuneEntries.size() / config.batchSize;
    if (numBatches == 0)
        throw std::runtime_error("training dataset has fewer accepted entries than one batch");

    // Search a new K or use previous K. Policy-only tuning does not use a
    // value scaling factor and must not scan the corpus for calibration.
    Float K = Float(1.0) / Evaluation::ScalingFactor;
    if (!config.tuneEval) {
        MESSAGEL("Skip scaling factor search for policy-only tuning.");
    }
    else if (config.usePreviousScalingFactor) {
        MESSAGEL("Use previous inv scaling factor = " << K);
    }
    else {
        MESSAGEL("Start seaching for optimal inv scaling factor...");
        K = searchOptimalInvScalingFactor();
    }

    // Init gradient array and optimizer
    std::vector<TuneGradient> gradients(tuneParams.size());
    AdamOptimizer<TuneParam>  optim(tuneParams.size(),
                                   TuneParam(config.learningRate),
                                   TuneParam(config.weightDecay));

    MESSAGEL("Start training for " << epochs << " epochs, lr = " << optim.currentLR()
                                   << ", batch size = " << config.batchSize
                                   << ", number of batches = " << numBatches << ".");

    for (size_t epoch = 0; epoch <= epochs; epoch++) {
        Time startTime = now();

        for (size_t batch = 0; epoch > 0 && batch < numBatches; batch++) {
            // Zero out all gradients
            std::fill(gradients.begin(), gradients.end(), Float(0));

            // Compute gradient of all parameters using current K
            computeGradientBatch(gradients, K, batch);
            if (std::any_of(gradients.begin(), gradients.end(), [](TuneGradient gradient) {
                    return !std::isfinite(gradient);
                }))
                throw std::runtime_error("non-finite gradient in tuning batch");

            // Update parameters with gradient using optimizer
            optim.step(tuneParams, gradients);
        }

        // Print out current epoch and loss
        Float valueLoss     = computeEvaluationLoss(K, false);
        Float policyLoss    = computeMoveScoreLoss(false);
        Float valueValLoss  = computeEvaluationLoss(K, true);
        Float policyValLoss = computeMoveScoreLoss(true);
        if (!(std::isfinite(valueLoss) && std::isfinite(policyLoss)
              && std::isfinite(valueValLoss) && std::isfinite(policyValLoss)))
            throw std::runtime_error("non-finite tuning metric");
        Time  elapsed       = now() - startTime;
        if (!valTuneEntries.empty())
            MESSAGEL("Epoch " << epoch << " | Value " << valueLoss << " | Policy " << policyLoss
                              << " | ValueVal " << valueValLoss << " | PolicyVal " << policyValLoss
                              << " | Time(ms) " << elapsed);
        else
            MESSAGEL("Epoch " << epoch << " | Value " << valueLoss << " | Policy " << policyLoss
                              << " | Time(ms) " << elapsed);

        // Call callback after each epoch completed
        if (callback) {
            TuningStatistic stat;
            stat.currentEpoch   = epoch;
            stat.valueLoss      = valueLoss;
            stat.policyLoss     = policyLoss;
            stat.valueValLoss   = valueValLoss;
            stat.policyValLoss  = policyValLoss;
            stat.elapsedSeconds = double(elapsed) / 1000.0;
            stat.scalingFactor  = 1.0 / double(K);
            callback(stat);
        }

        // Recompute K for tuned evaluation
        if (config.tuneEval && !config.usePreviousScalingFactor && epoch > 0
            && config.recomputeInterval
            && epoch % config.recomputeInterval == 0) {
            K = searchOptimalInvScalingFactor();
        }
    }

    Time totalElapsed = now() - initTime;
    MESSAGEL("Training completed in " << (totalElapsed / 1000) << " seconds.");
}

/// initParams() inits tuneParams according to their value in the live
/// Evaluation:: model tables. It also associates TuneParam index with its
/// table address. Parameters loaded from the tables will be automatically
/// saved back when Tuner is destroyed.
void Tuner::initParams()
{
    std::vector<int>                       ruleSetIdx;
    PRNG                                   prng = PRNG(domainSeed(config.seed, 0x706172616d2d696eULL));
    std::uniform_real_distribution<double> rand;

    if (config.tuneRule[FREESTYLE])
        ruleSetIdx.push_back(FREESTYLE);
    if (config.tuneRule[STANDARD])
        ruleSetIdx.push_back(STANDARD);
    if (config.tuneRule[RENJU]) {
        ruleSetIdx.push_back(RENJU + BLACK);
        ruleSetIdx.push_back(RENJU + WHITE);
    }

    // The tuner intentionally mutates the LIVE Evaluation:: tables in place
    // through these stored addresses: evaluation during tuning always sees the
    // current candidate parameters without a copy/swap step.
    for (int r : ruleSetIdx) {
        if (config.tuneEval) {
            addArrayParams<Eval>(
                Evaluation::EVALS[r],
                [](const Eval &ev, size_t) { return TuneParam(ev); },
                [](Eval &ev, size_t, TuneParam param) {
                    constexpr Float EvalMin = Float(std::numeric_limits<Eval>::min());
                    constexpr Float EvalMax = Float(std::numeric_limits<Eval>::max());
                    ev = static_cast<Eval>(std::clamp(Float(param), EvalMin, EvalMax));
                });

            addArrayParams<Eval>(
                Evaluation::EVALS_THREAT[r],
                [](const Eval &ev, size_t) { return TuneParam(ev); },
                [](Eval &ev, size_t, TuneParam param) {
                    constexpr Float EvalMin = Float(std::numeric_limits<Eval>::min());
                    constexpr Float EvalMax = Float(std::numeric_limits<Eval>::max());
                    ev = static_cast<Eval>(std::clamp(Float(param), EvalMin, EvalMax));
                });
        }

        if (config.tuneMoveScore) {
            addArrayParams<MoveScorePair, arraySize(Evaluation::P4SCORES[0]), 2>(
                Evaluation::P4SCORES[r],
                [scale      = config.moveScoreScale,
                 bias       = config.moveScoreBias,
                 randomInit = config.randomMoveScoreInit,
                 &rand,
                 &prng](const MoveScorePair &scorePair, size_t offset) {
                    Score score = scorePair[offset];
                    return randomInit ? TuneParam(rand(prng))
                                      : encodeIntegerForTruncatingExport(score, scale, bias);
                },
                [scoreMin = (Float)config.moveScoreMin,
                 scoreMax = (Float)config.moveScoreMax,
                 scale    = config.moveScoreScale,
                 bias =
                     config.moveScoreBias](MoveScorePair &scorePair, size_t offset, TuneParam param) {
                    Float score = Float(param) * scale + bias;
                    scorePair[offset] =
                        static_cast<Score>(std::clamp(score, scoreMin, scoreMax));
                });
        }
    }

    MESSAGEL(tuneParams.size() << " parameters initialized.");
}

/// saveParams() saves tuneParams back to their associated config value
void Tuner::saveParams() const
{
    for (const ParamsSyncRecord &record : syncRecords) {
        assert(tuneParams.size() >= record.baseIndex + record.numElems * record.paramPerElem);

        for (size_t i = 0; i < record.numElems; i++)
            for (size_t j = 0; j < record.paramPerElem; j++)
                record.setter(record[i],
                              j,
                              tuneParams[record.baseIndex + i * record.paramPerElem + j]);
    }

    MESSAGEL(tuneParams.size() << " parameters saved.");
}

void Tuner::appendTuneSample(PreparedCorpus &tuneEntries,
                             const Board    &board,
                             Rule            rule,
                             uint8_t         resultTimesTwo,
                             Pos             bestMove,
                             CompileScratch &scratch) const
{
    Value staticEval = Evaluation::evaluate(board, rule);
    if (staticEval < INT16_MIN || staticEval > INT16_MAX)
        throw std::overflow_error("static evaluation exceeds int16 storage");

    scratch.evalTerms.clear();
    if (config.tuneEval) {
        Evaluation::EvalInfo evalInfo(board, rule);
        collectEvalCoeffs(rule, evalInfo, [this, &scratch](int coeff, int coeffScale, void *addr) {
            if (coeff == 0)
                return;
            int scaledCoeff = int(coeff * CoeffScale) / coeffScale;
            if (scaledCoeff < INT16_MIN || scaledCoeff > INT16_MAX)
                throw std::overflow_error("value coefficient exceeds int16 storage");
            scratch.evalTerms.push_back(
                {static_cast<int16_t>(scaledCoeff), paramIndex(addr)});
        });
    }

    scratch.policyCandidates.clear();
    uint16_t bestCandidate = PreparedCorpus::NoPolicyTarget;
    if (config.tuneMoveScore && bestMove != Pos {board.size(), board.size()}
        && bestMove != Pos::NONE && bestMove != Pos::PASS && board.isEmptyCandidate(bestMove)) {
        collectMoveScoreCoeffs(
            rule,
            board,
            [this, bestMove, &scratch, &bestCandidate](Pos   pos,
                                                       int   coeffSelf,
                                                       int   coeffOppo,
                                                       void *addrSelf,
                                                       void *addrOppo) {
                if (coeffSelf != 1 || coeffOppo != 1)
                    throw std::logic_error("compact policy storage requires unit coefficients");

                if (pos == bestMove)
                    bestCandidate = static_cast<uint16_t>(scratch.policyCandidates.size());
                PolicyCandidate candidate;
                candidate.indices[0] = paramIndex(addrSelf, 0);
                candidate.indices[1] = paramIndex(addrOppo, 1);
                scratch.policyCandidates.push_back(candidate);
            });
        if (bestCandidate == PreparedCorpus::NoPolicyTarget)
            throw std::logic_error("best move is missing from policy candidates");
    }

    if (config.tuneEval) {
        Float linearEval = 0;
        for (const TuneCoeff &term : scratch.evalTerms)
            linearEval += term.coeff * tuneParams[term.index];
        linearEval /= CoeffScale;
        if (!checkEqual(Float(staticEval), linearEval))
            throw std::logic_error("prepared linear evaluation differs from engine evaluation");
    }

    tuneEntries.append(resultTimesTwo,
                       static_cast<int16_t>(staticEval),
                       scratch.evalTerms,
                       scratch.policyCandidates,
                       bestCandidate);
}

/// initTuneEntries() inits tuneEntries from dataEntry read from datasets.
/// DataEntry that does not satisfy a certain condition will be skipped.
void Tuner::initTuneEntries(PreparedCorpus &tuneEntries,
                            class Dataset  &dataset,
                            bool            buildShuffleOrder)
{
    tuneEntries.clear();

    std::deque<std::future<PreparedCorpus>> jobs;
    const size_t maxPendingJobs = std::max<size_t>(threadPool.get_thread_count(), 1);

    auto collectFrontJob = [&]() {
        PreparedCorpus fragment = jobs.front().get();
        jobs.pop_front();
        tuneEntries.append(std::move(fragment));
    };

    // Read dataset and convert bounded batches to compact corpus fragments.
    size_t totalEntriesRead = 0;
    if (dataset.supportsGames()) {
        using GameWork = std::pair<GameEntry, size_t>;
        bool reachedEnd = false;
        while (totalEntriesRead < config.maxTuneEntries && !reachedEnd) {
            std::vector<GameWork> games;
            size_t                chunkEntries = 0;

            while (chunkEntries < config.batchSize
                   && totalEntriesRead + chunkEntries < config.maxTuneEntries) {
                GameEntry game;
                if (!dataset.nextGame(&game)) {
                    reachedEnd = true;
                    break;
                }

                size_t gameMoveCount = game.moveSequence.size();
                size_t remaining = config.maxTuneEntries - totalEntriesRead - chunkEntries;
                size_t moveLimit = std::min(gameMoveCount, remaining);
                if (moveLimit != 0) {
                    chunkEntries += moveLimit;
                    games.emplace_back(std::move(game), moveLimit);
                }
                if (moveLimit < gameMoveCount)
                    break;
            }

            if (games.empty())
                continue;

            totalEntriesRead += chunkEntries;
            auto sharedGames = std::make_shared<std::vector<GameWork>>(std::move(games));
            jobs.emplace_back(threadPool.submit_task(
                [this, games = std::move(sharedGames), chunkEntries]() -> PreparedCorpus {
                    PreparedCorpus entries;
                    CompileScratch scratch;
                    entries.reserveSamples(chunkEntries);

                    for (const auto &work : *games) {
                        const GameEntry &game      = work.first;
                        size_t           moveLimit = work.second;
                        Board            board(game.boardsize);
                        board.newGame(game.rule);
                        for (Pos pos : game.initPosition)
                            board.move(game.rule, pos);

                        for (size_t moveIndex = 0; moveIndex < moveLimit; moveIndex++) {
                            size_t ply = game.initPosition.size() + moveIndex;
                            if (config.tuneRule[game.rule]
                                && game.boardsize >= config.boardSizeMin
                                && game.boardsize <= config.boardSizeMax && ply >= config.minPly
                                && ply + config.minPlyBeforeFull
                                       <= int(game.boardsize) * int(game.boardsize)) {
                                Result result = board.sideToMove() == WHITE
                                                    ? game.result
                                                    : flipResult(game.result);
                                uint8_t resultTimesTwo = result == RESULT_WIN    ? 2
                                                         : result == RESULT_DRAW ? 1
                                                                                 : 0;
                                appendTuneSample(entries,
                                                 board,
                                                 game.rule,
                                                 resultTimesTwo,
                                                 game.moveSequence[moveIndex].move,
                                                 scratch);
                            }
                            board.move(game.rule, game.moveSequence[moveIndex].move);
                        }
                    }
                    return entries;
                }));

            if (jobs.size() >= maxPendingJobs)
                collectFrontJob();
        }
    }
    else {
        while (totalEntriesRead < config.maxTuneEntries) {
            size_t entriesToRead =
                std::min(config.batchSize, config.maxTuneEntries - totalEntriesRead);
            std::vector<DataEntry> dataEntries;
            dataEntries.reserve(entriesToRead);
            for (size_t i = 0; i < entriesToRead; i++) {
                DataEntry dataEntry;
                if (!dataset.next(&dataEntry))
                    break;
                dataEntries.push_back(std::move(dataEntry));
            }

            if (dataEntries.empty())
                break;
            totalEntriesRead += dataEntries.size();

            jobs.emplace_back(threadPool.submit_task(
                [this, data = std::move(dataEntries)]() -> PreparedCorpus {
                    std::unordered_map<int, Board> boardObjectCache;
                    PreparedCorpus                 entries;
                    CompileScratch                 scratch;
                    entries.reserveSamples(data.size());

                    for (const DataEntry &dataEntry : data) {
                        if (!config.tuneRule[dataEntry.rule]
                            || dataEntry.boardsize < config.boardSizeMin
                            || dataEntry.boardsize > config.boardSizeMax
                            || dataEntry.position.size() < config.minPly
                            || dataEntry.position.size() + config.minPlyBeforeFull
                                   > int(dataEntry.boardsize) * int(dataEntry.boardsize))
                            continue;

                        auto boardIt = boardObjectCache.find(dataEntry.boardsize);
                        if (boardIt == boardObjectCache.end()) {
                            boardIt = boardObjectCache
                                          .emplace(std::piecewise_construct,
                                                   std::forward_as_tuple(dataEntry.boardsize),
                                                   std::forward_as_tuple(dataEntry.boardsize))
                                          .first;
                        }

                        Board &board = boardIt->second;
                        board.newGame(dataEntry.rule);
                        for (Pos pos : dataEntry.position)
                            board.move(dataEntry.rule, pos);

                        uint8_t resultTimesTwo = dataEntry.result == RESULT_WIN    ? 2
                                                 : dataEntry.result == RESULT_DRAW ? 1
                                                                                   : 0;
                        appendTuneSample(entries,
                                         board,
                                         dataEntry.rule,
                                         resultTimesTwo,
                                         dataEntry.move,
                                         scratch);
                    }

                    return entries;
                }));

            if (jobs.size() >= maxPendingJobs)
                collectFrontJob();
        }
    }

    MESSAGEL("Read " << totalEntriesRead << " tune entries from dataset, initializing...");

    while (!jobs.empty())
        collectFrontJob();

    MESSAGEL(tuneEntries.size() << " tune entries initialized in "
                                << tuneEntries.capacityBytes() / (1024.0 * 1024.0) << " MiB.");

    if (buildShuffleOrder && config.shuffleTuneEntries) {
        MESSAGEL("Creating logical shuffle order...");

        if (tuneEntries.size() > std::numeric_limits<uint32_t>::max())
            throw std::length_error("shuffle order exceeds 32-bit sample indices");
        trainSampleOrder.resize(tuneEntries.size());
        std::iota(trainSampleOrder.begin(), trainSampleOrder.end(), uint32_t(0));
        PRNG prng(domainSeed(config.seed, 0x73687566666c6500ULL));
        std::shuffle(trainSampleOrder.begin(), trainSampleOrder.end(), prng);
    }
}

/// searchOptimalInvScalingFactor() searches the optimal K in formula:
/// sigma = 1 + (1 / exp(-Eval * K)). by brute-forcely steps through the whole
/// scaling factor space for many iterations and finds the point that minimize
/// error of the current static evaluation and win rate in target tune entries.
Float Tuner::searchOptimalInvScalingFactor() const
{
    assert(config.nStepsPerIteration);

    Float startK = 1.0 / Float(config.scalingFactorMin);
    Float endK   = 1.0 / Float(config.scalingFactorMax);
    Float stepK  = (endK - startK) / Float(config.nStepsPerIteration);
    Float bestK  = 0;

    for (int iter = 1; iter <= config.nIterations; iter++) {
        Float k        = startK;
        Float bestLoss = std::numeric_limits<Float>::max();

        for (int i = 0; i < config.nStepsPerIteration; i++) {
            Float loss = computeEvaluationLoss<false>(k, false);

            if (loss < bestLoss) {
                bestLoss = loss;
                bestK    = k;
            }

            k += stepK;
        }

        MESSAGEL("Iteration " << iter << " | K " << bestK << " | Loss " << bestLoss);

        startK = bestK - stepK;
        endK   = bestK + stepK;
        stepK  = stepK * 2 / Float(config.nStepsPerIteration);
    }

    MESSAGEL("Optimal inv scaling factor K = " << bestK << " after " << config.nIterations
                                               << " iteration.");

    return bestK;
}

/// computeEvaluationLoss() computes loss between the current tuned/static
/// evaluation and target win rate in all tune entries using the given K.
template <bool UseTunedEval>
Float Tuner::computeEvaluationLoss(Float K, bool validation) const
{
    if (!config.tuneEval)
        return Float(0.0);

    const PreparedCorpus &entries = validation ? valTuneEntries : trainTuneEntries;

    if (entries.empty())
        return Float(0.0);

    return parallelIndexReduce<Float>(threadPool,
                                      entries.size(),
                                      Float(0.0),
                                      [this, &entries, K, validation](size_t logicalSample) {
                                          size_t sample =
                                              !validation && !trainSampleOrder.empty()
                                                  ? trainSampleOrder[logicalSample]
                                                  : logicalSample;
                                          return ::computeEvalLoss<UseTunedEval>(
                                              entries, sample, tuneParams, K, config.lossType);
                                      })
           / Float(entries.size());
}

/// computeMoveScoreLoss() computes loss of current move scores between
/// the target best move in all tune entries.
Float Tuner::computeMoveScoreLoss(bool validation) const
{
    if (!config.tuneMoveScore)
        return Float(0.0);

    const PreparedCorpus &entries = validation ? valTuneEntries : trainTuneEntries;

    if (entries.empty())
        return Float(0.0);

    return parallelIndexReduce<Float>(threadPool,
                                      entries.size(),
                                      Float(0.0),
                                      [this, &entries, validation](size_t logicalSample) {
                                          size_t sample =
                                              !validation && !trainSampleOrder.empty()
                                                  ? trainSampleOrder[logicalSample]
                                                  : logicalSample;
                                          return ::computeMoveScoreLoss(entries,
                                                                        sample,
                                                                        tuneParams,
                                                                        config.moveScoreLossGamma);
                                      })
           / Float(entries.size());
}

/// computeGradients() computes gradients of all parameters used in one tune
/// entries batch and accumulates them into gradients vector. These gradients
/// then will be used to tune the parameters with a gradient descent optimizer.
void Tuner::computeGradientBatch(std::vector<TuneGradient> &grads, Float K, size_t batchIdx)
{
    assert(grads.size() == tuneParams.size());
    const size_t     numJobs       = std::min(LogicalPartitions, config.batchSize);
    const size_t     baseJobSize   = config.batchSize / numJobs;
    const size_t     remainder     = config.batchSize % numJobs;

    if (partitionGradients.size() != numJobs)
        partitionGradients.assign(numJobs, std::vector<TuneGradient>(tuneParams.size()));

    std::vector<std::future<void>> gradJobs;
    gradJobs.reserve(numJobs);
    size_t batchBegin = batchIdx * config.batchSize;

    for (size_t jobIdx = 0; jobIdx < numJobs; jobIdx++) {
        // Get range of tune entries for this job
        size_t jobOffset = jobIdx * baseJobSize + std::min(jobIdx, remainder);
        size_t jobSize   = baseJobSize + (jobIdx < remainder);
        size_t jobBegin = batchBegin + jobOffset;
        size_t jobEnd   = jobBegin + jobSize;

        // Accumulate local gradient asynchronously
        auto job = threadPool.submit_task([this, K, jobIdx, jobBegin, jobEnd] {
            std::vector<TuneGradient> &localGrads = partitionGradients[jobIdx];
            std::fill(localGrads.begin(), localGrads.end(), TuneGradient(0));

            for (size_t logicalSample = jobBegin; logicalSample < jobEnd; logicalSample++) {
                size_t sample = trainSampleOrder.empty() ? logicalSample
                                                         : trainSampleOrder[logicalSample];
                ::computeEvalGradient(trainTuneEntries,
                                      sample,
                                      localGrads,
                                      tuneParams,
                                      K,
                                      config.lossType);
                ::computeMoveScoreGradient(trainTuneEntries,
                                           sample,
                                           localGrads,
                                           tuneParams,
                                           config.moveScoreLossGamma);
            }

            // Scale gradient according to batch size
            TuneGradient scale = TuneGradient(1 / Float(config.batchSize));
            for (TuneGradient &gradient : localGrads)
                gradient *= scale;

        });
        gradJobs.push_back(std::move(job));
    }

    for (auto &job : gradJobs)
        job.get();

    // Reduce in logical partition order, independent of physical scheduling.
    for (const std::vector<TuneGradient> &localGrads : partitionGradients) {
        assert(grads.size() == localGrads.size());

        for (size_t i = 0; i < grads.size(); i++)
            grads[i] += localGrads[i];
    }
}

/// addParams() adds a continous range of params in config to tuneParams
void Tuner::addParams(void         *address,
                      size_t        numElems,
                      uint32_t      elemSize,
                      uint32_t      paramPerElem,
                      ParamGetter<> getter,
                      ParamSetter<> setter)
{
    assert(paramPerElem > 0);
    size_t baseIndex = tuneParams.size();

    // Init parameters from getter and add them to tuneParams
    if (numElems > std::numeric_limits<size_t>::max() / paramPerElem)
        throw std::length_error("tuning parameter count overflows size_t");
    size_t numParams = numElems * paramPerElem;
    constexpr size_t MaxParameterCount = size_t(std::numeric_limits<ParameterId>::max()) + 1;
    if (numParams > MaxParameterCount - baseIndex)
        throw std::length_error("tuning parameter count exceeds ParameterId capacity");

    syncRecords.push_back(ParamsSyncRecord {baseIndex,
                                            numElems,
                                            elemSize,
                                            paramPerElem,
                                            address,
                                            std::move(getter),
                                            std::move(setter)});
    const ParamsSyncRecord &record = syncRecords.back();

    tuneParams.reserve(baseIndex + numParams);
    for (size_t i = 0; i < numElems; i++) {
        size_t elementBase = baseIndex + i * paramPerElem;
        auto [it, inserted] = paramIndices.emplace(
            record[i],
            ParameterAddress {static_cast<ParameterId>(elementBase), paramPerElem});
        if (!inserted)
            throw std::logic_error("duplicate tuning parameter address");

        for (size_t j = 0; j < paramPerElem; j++) {
            TuneParam param = record.getter(record[i], j);
            if (!std::isfinite(param))
                throw std::runtime_error("non-finite initial tuning parameter");
            tuneParams.emplace_back(param);
        }
    }
}

/// paramIndex() finds tuneParams index according to address of its config value
ParameterId Tuner::paramIndex(const void *addr, size_t offset) const
{
    auto it = paramIndices.find(addr);
    if (it == paramIndices.end())
        throw std::logic_error("unknown tuning parameter address");

    if (offset >= it->second.parameterCount)
        throw std::out_of_range("tuning parameter offset is out of range");
    size_t index = size_t(it->second.baseIndex) + offset;
    if (index >= tuneParams.size())
        throw std::out_of_range("tuning parameter offset is out of range");
    return static_cast<ParameterId>(index);
}

}  // namespace Tuning
