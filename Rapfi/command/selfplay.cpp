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

#include "../config.h"
#include "../core/iohelper.h"
#include "../core/utils.h"
#include "../search/hashtable.h"
#include "../search/searchconfig.h"
#include "../search/searchthread.h"
#include "../tuning/datawriter.h"
#include "argutils.h"
#include "command.h"

#define CXXOPTS_NO_REGEX
#include <csignal>
#include <cxxopts.hpp>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>

static std::function<void(int)> signalFunc;
static void                     setupSignalHandler(std::function<void(int)> handler)
{
    signalFunc         = std::move(handler);
    auto signalHandler = [](int signal) {
        if (signalFunc)
            signalFunc(signal);
    };

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    std::signal(SIGSEGV, signalHandler);
    std::signal(SIGILL, signalHandler);
    std::signal(SIGABRT, signalHandler);
    std::signal(SIGFPE, signalHandler);
#ifdef SIGHUP
    std::signal(SIGHUP, signalHandler);
#endif
#ifdef SIGQUIT
    std::signal(SIGQUIT, signalHandler);
#endif
}

using namespace Tuning;

namespace {

std::vector<std::vector<Pos>> readOpenings(std::istream &is, int boardSize)
{
    std::vector<std::vector<Pos>> ops;
    std::string                   opStr;
    size_t                        lineCount = 0;

    while (std::getline(is, opStr)) {
        lineCount++;
        if (opStr.empty())
            continue;

        try {
            ops.push_back(Command::parsePositionString(opStr, boardSize, boardSize));
        }
        catch (const std::exception &e) {
            throw std::runtime_error("illegal opening [line " + std::to_string(lineCount)
                                     + "]: " + e.what());
        }
    }

    return ops;
}

/// All knobs of one selfplay run, parsed from the command line.
struct SelfplayConfig
{
    size_t                        numGames;
    size_t                        numThreads;
    size_t                        hashSizeMb;
    int                           boardSizeMin, boardSizeMax;
    Rule                          rule;
    int                           multipv, multipvDecaySteps;
    double                        meanNodes, varNodes, nodesDecay;
    int                           maxDecaySteps;
    uint64_t                      minNodes;
    int                           matePly;
    int                           drawValue, drawCount, minDrawPly;
    int                           forceDrawPly, forceDrawPlyLeft;
    Time                          reportInterval;
    bool                          noMateMultiPV;
    bool                          generateOpening;
    bool                          silence;
    std::vector<std::vector<Pos>> openings;
    Opening::OpeningGenConfig     opengenCfg;
    Command::DataWriterType       dataWriterType;
    std::string                   outputPath;
};

/// Put an opening on the fresh `board`: either an auto-generated (balanced) opening,
/// or a random book opening under a random symmetry transform. No-op when neither
/// source is configured (game starts from the empty board).
void applyOpening(Board &board, const SelfplayConfig &cfg, PRNG &prng)
{
    if (cfg.generateOpening) {
        Opening::OpeningGenerator og(board.size(), cfg.rule, cfg.opengenCfg, prng);

        // Generate a valid opening
        for (;;) {
            bool balanced = og.next();

            // If we want to generate a balanced opening, abandon those not balanced
            if (!balanced && cfg.opengenCfg.balanceWindow > 0
                && (cfg.opengenCfg.balance1Nodes > 0 || cfg.opengenCfg.balance2Nodes > 0))
                continue;
            else
                break;
        }

        // Put opening
        for (int i = 0; i < og.getBoard().ply(); i++) {
            board.move(cfg.rule, og.getBoard().getHistoryMove(i));
        }

        if (!cfg.silence)
            MESSAGEL("Put generated opening " << og.positionString());
    }
    else if (!cfg.openings.empty()) {
        std::uniform_int_distribution<size_t> openingDis(0, cfg.openings.size() - 1);
        std::uniform_int_distribution<int>    transformDis(0, TRANS_NB - 1);
        size_t                                openingIdx = openingDis(prng);
        TransformType                         transform  = TransformType(transformDis(prng));

        // Apply opening pos
        for (Pos pos : cfg.openings[openingIdx]) {
            Pos transformedPos = applyTransform(pos, board.size(), transform);
            board.move(cfg.rule, transformedPos);
        }
    }
}

/// Play one selfplay game on the prepared board (opening already applied) until the
/// win/loss/draw adjudication triggers, and return the recorded game entry.
GameEntry playOneGame(Board                            &board,
                      const SelfplayConfig             &cfg,
                      PRNG                             &prng,
                      std::normal_distribution<double> &nodesDis)
{
    // Set search options and init
    Search::SearchOptions options;
    options.rule                = {cfg.rule, GameRule::FREEOPEN};
    options.multiPV             = cfg.multipv;
    options.balanceMode         = Search::SearchOptions::BALANCE_NONE;
    options.disableOpeningQuery = true;
    Search::Engine.clear(true);

    // Setup game entry data
    GameEntry gameEntry;
    for (int i = 0; i < board.ply(); i++)
        gameEntry.initPosition.push_back(board.getHistoryMove(i));
    gameEntry.boardsize = board.size();
    gameEntry.rule      = cfg.rule;

    // Selfplay loop
    Value searchValue = VALUE_ZERO;
    int   drawCnt     = 0;
    while (board.movesLeft() > 0) {
        // Stop self-play game if force draw or board is full
        if ((cfg.forceDrawPly && board.ply() >= cfg.forceDrawPly)
            || (cfg.forceDrawPlyLeft && board.movesLeft() <= cfg.forceDrawPlyLeft))
            break;

        // Set search limits, make sure max nodes stays in [0, +inf)
        int    steps      = board.ply() - (int)gameEntry.initPosition.size();
        double nodesScale = std::pow(cfg.nodesDecay, std::min(steps, cfg.maxDecaySteps));
        options.maxNodes  = std::max<uint64_t>((uint64_t)std::max(nodesDis(prng) * nodesScale, 0.0),
                                              cfg.minNodes);
        if (cfg.multipvDecaySteps > 0 && steps > 0 && steps % cfg.multipvDecaySteps == 0)
            options.multiPV = std::max(1, options.multiPV - 1);

        // Start thinking and wait for finish
        Search::Engine.startThinking(board, options);
        Search::Engine.waitForIdle();
        auto mainThread = Search::Engine.main();

        // We might have no legal move in Renju mode, which is regarded as loss
        if (mainThread->rootMoves.empty()) {
            searchValue = mated_in(0);
            break;
        }

        // Record best move result
        searchValue  = mainThread->rootMoves[0].value;
        Pos bestMove = Search::Engine.ctx.bestMove;
        gameEntry.moveSequence.push_back({bestMove, Eval(searchValue)});
        // A multipv search can still yield a single root move (a forced defence, or
        // renju forbidden points leaving one legal move). Such a ply must be recorded
        // as a plain move: MULTIPV_BEGIN + 1 - 2 would underflow the tag into
        // POLICY_ARRAY_INT16, making the data writer read a policy array that was
        // never allocated.
        int numPVMoves = std::min<int>(options.multiPV, mainThread->rootMoves.size());
        if (options.multiPV > 1 && numPVMoves > 1) {
            auto &moveData = gameEntry.moveSequence.back();
            moveData.tag  = DataEntry::MoveDataTag(DataEntry::MULTIPV_BEGIN + numPVMoves - 2);
            moveData.multiPvMoves = new PVMove[numPVMoves - 1];
            for (int i = 1; i < numPVMoves; i++) {
                auto &rm = mainThread->rootMoves[i];
                assert(rm.pv[0] != bestMove);
                moveData.multiPvMoves[i - 1] = {
                    rm.pv[0],
                    Eval(rm.value != VALUE_NONE ? rm.value : rm.previousValue)};
            }
        }

        // Stop self-play game if win/loss is found
        if (std::abs(searchValue) >= mate_in(cfg.matePly))
            break;
        if (cfg.noMateMultiPV && std::abs(searchValue) >= VALUE_MATE_IN_MAX_PLY)
            options.multiPV = 1;

        // Stop self-play game if draw adjudication
        if (cfg.drawCount && std::abs(searchValue) <= cfg.drawValue) {
            if (++drawCnt >= cfg.drawCount && board.ply() >= cfg.minDrawPly)
                break;
        }
        else
            drawCnt = 0;

        // Make the move
        board.move(cfg.rule, bestMove);
    }

    // Save game result (root samples)
    Result result    = searchValue >= VALUE_MATE_IN_MAX_PLY    ? RESULT_WIN
                       : searchValue <= VALUE_MATED_IN_MAX_PLY ? RESULT_LOSS
                                                               : RESULT_DRAW;
    // `result` is from the final side to move's pov; GameEntry::result is white pov.
    gameEntry.result = board.sideToMove() == WHITE ? result : flipResult(result);
    return gameEntry;
}

/// Parse the selfplay command line into a SelfplayConfig (reading the opening book
/// file if one is given). Exits after printing usage on --help or argument errors.
SelfplayConfig parseSelfplayArguments(int argc, char *argv[])
{
    using namespace Command;

    SelfplayConfig cfg;

    cxxopts::Options options("rapfi selfplay");
    options.add_options()  //
        ("o,output",
         "Save data entries of played games to a binary file",
         cxxopts::value<std::string>())  //
        ("output-type",
         "Output dataset type, one of [txt, bin, bin_lz4, binpack, binpack_lz4]",
         cxxopts::value<std::string>()->default_value("binpack_lz4"))             //
        ("n,number", "Number of games to play", cxxopts::value<size_t>())         //
        ("boardsize-min", "Minimal board size in [5,22]", cxxopts::value<int>())  //
        ("boardsize-max", "Maximal board size in [5,22]", cxxopts::value<int>())  //
        ("opening",
         "Path to the opening book file. If not specified, auto generated openings are used",
         cxxopts::value<std::string>())  //
        ("multipv",
         "The maximum number of multipv to record (must be at least 1)",
         cxxopts::value<int>()->default_value("1"))  //
        ("multipv-decay-steps",
         "The number of steps to decrease multipv by 1 (0 for not enabled)",
         cxxopts::value<int>()->default_value("0"))  //
        ("no-multipv-after-mate",
         "Disable multipv after mate has been found")  //
        ("mean-nodes",
         "Mean of a normal distribution of num search nodes",
         cxxopts::value<double>()->default_value("100000"))  //
        ("var-nodes",
         "Variance of a normal distribution of num search nodes",
         cxxopts::value<double>()->default_value("10000"))  //
        ("nodes-decay",
         "The lambda to decay the number of search nodes",
         cxxopts::value<double>()->default_value("1.0"))  //
        ("max-nodes-decay-steps",
         "The maximum number of steps to decay the number of search nodes",
         cxxopts::value<int>()->default_value("100"))  //
        ("min-nodes",
         "Minimal number of nodes for search",
         cxxopts::value<uint64_t>()->default_value("20000"))  //
        ("mate-ply",
         "Judge win/loss if there is only mate-ply before mate (must be at least 1)",
         cxxopts::value<int>()->default_value("1"))  //
        ("draw-count",
         "Draw if |value| <= draw-value occurred for [draw-count] consecutive moves, and at least "
         "min-draw-ply stones are on board (0 for not enabled)",
         cxxopts::value<int>()->default_value("7"))  //
        ("draw-value",
         "Draw if |value| <= [draw-value] occurred for draw-count consecutive moves, and at least "
         "min-draw-ply stones are on board",
         cxxopts::value<int>()->default_value("10"))  //
        ("min-draw-ply",
         "Draw if |value| <= draw-value occurred for draw-count consecutive moves, and at least "
         "[min-draw-ply] stones are on board",
         cxxopts::value<int>()->default_value("100"))  //
        ("force-draw-ply",
         "Force draw after this ply (0 for not enabled)",
         cxxopts::value<int>()->default_value("0"))  //
        ("force-draw-plyleft",
         "Force draw when left ply is less than this",
         cxxopts::value<int>()->default_value("0"))  //
        ("report-interval",
         "Time (ms) between two progress report message",
         cxxopts::value<Time>()->default_value("60000"))  //
        ("h,help", "Print selfplay usage");
    addPlayOptions(options);
    addOpengenOptions(options, cfg.opengenCfg);

    parseSubcommandArguments(options, argc, argv, "selfplay argument",
                             [&](const cxxopts::ParseResult &args) {

        if (args.count("output")) {
            // Open output file and change output stream
            cfg.outputPath     = args["output"].as<std::string>();
            cfg.dataWriterType = parseDataWriterType(args["output-type"].as<std::string>());
        }

        if (args.count("boardsize-min") || args.count("boardsize-max")) {
            cfg.boardSizeMin = args["boardsize-min"].as<int>();
            cfg.boardSizeMax = args["boardsize-max"].as<int>();
            if (cfg.boardSizeMin > cfg.boardSizeMax)
                throw std::invalid_argument("invalid board size range");
        }
        else {
            cfg.boardSizeMin = cfg.boardSizeMax = args["boardsize"].as<int>();
        }
        if (cfg.boardSizeMin < 5 || cfg.boardSizeMax > 22)
            throw std::invalid_argument("board size must in range [5,22]");

        if (args.count("opening")) {
            std::string filename = args["opening"].as<std::string>();
            if (cfg.boardSizeMin != cfg.boardSizeMax)
                throw std::invalid_argument(
                    "opening file can only be used with constant board size");

            // Open output file and change output stream
            std::ifstream openingFile(filename);
            if (!openingFile.is_open())
                throw std::invalid_argument("unable to open opening file " + filename);

            cfg.openings        = readOpenings(openingFile, cfg.boardSizeMin);
            cfg.generateOpening = false;
        }
        else {
            cfg.opengenCfg      = parseOpengenConfig(args);
            cfg.generateOpening = true;
        }

        cfg.numGames          = args["number"].as<size_t>();
        cfg.rule              = parseRule(args["rule"].as<std::string>());
        cfg.numThreads        = std::max<size_t>(args["thread"].as<size_t>(), 1);
        cfg.hashSizeMb        = std::max<size_t>(args["hashsize"].as<size_t>(), 1);
        cfg.multipv           = std::max(args["multipv"].as<int>(), 1);
        cfg.multipvDecaySteps = std::max(args["multipv-decay-steps"].as<int>(), 0);
        cfg.noMateMultiPV     = args.count("no-multipv-after-mate");
        cfg.meanNodes         = std::max(args["mean-nodes"].as<double>(), 0.0);
        cfg.varNodes          = std::max(args["var-nodes"].as<double>(), 0.0);
        cfg.nodesDecay        = std::min(args["nodes-decay"].as<double>(), 1.0);
        cfg.maxDecaySteps     = std::max(args["max-nodes-decay-steps"].as<int>(), 0);
        cfg.minNodes          = args["min-nodes"].as<uint64_t>();
        cfg.matePly           = std::max(args["mate-ply"].as<int>(), 1);
        cfg.drawValue         = args["draw-value"].as<int>();
        cfg.drawCount         = args["draw-count"].as<int>();
        cfg.minDrawPly        = args["min-draw-ply"].as<int>();
        cfg.forceDrawPly      = args["force-draw-ply"].as<int>();
        cfg.forceDrawPlyLeft  = args["force-draw-plyleft"].as<int>();
        cfg.reportInterval    = args["report-interval"].as<Time>();
        cfg.silence           = args.count("no-search-message");

        if (cfg.meanNodes <= 0)
            throw std::invalid_argument("mean-nodes must be greater than 0");
        if (cfg.matePly < 1)
            throw std::invalid_argument("mate-ply must be at least 1");
        if (cfg.matePly >= MAX_MOVES)
            throw std::invalid_argument("mate-ply must be less than " + std::to_string(MAX_MOVES));
        if (cfg.drawValue < 0)
            throw std::invalid_argument("draw-value must be at least 0");
        if (cfg.drawCount < 0)
            throw std::invalid_argument("draw-count must be at least 0");
        if (cfg.minDrawPly < 0)
            throw std::invalid_argument("min-draw-ply must be at least 0");
        if (cfg.forceDrawPly < 0)
            throw std::invalid_argument("force-draw-ply must be at least 0");
    });

    return cfg;
}

}  // namespace

void Command::selfplay(int argc, char *argv[])
{
    SelfplayConfig              cfg = parseSelfplayArguments(argc, argv);
    std::unique_ptr<DataWriter> dataWriter;

    if (!cfg.outputPath.empty()) {
        // Create data writer
        if (cfg.dataWriterType == DataWriterType::Numpy) {
            ERRORL("Numpy data writer is not supported in selfplay.");
            std::exit(EXIT_FAILURE);
        }
        dataWriter = Tuning::makeDataWriter(cfg.dataWriterType, cfg.outputPath);
    }

    // Setup signal handler to close dataset file when receiving signal. Termination
    // requests exit cleanly; fault signals (SIGSEGV etc.) still flush the writer but
    // must report the crash and exit nonzero instead of masquerading as success.
    setupSignalHandler([&](int signal) {
        bool requested = signal == SIGINT || signal == SIGTERM;
        if (requested)
            MESSAGEL("Gracefully exiting...");
        else
            ERRORL("Terminated by signal " << signal << ", flushing dataset...");
        dataWriter.reset();
        std::exit(requested ? 0 : EXIT_FAILURE);
    });

    if (cfg.openings.size())
        MESSAGEL("Read " << cfg.openings.size() << " openings for selfplay.");
    else if (cfg.generateOpening)
        MESSAGEL("No opening file is specified, will use automatic opening generation.");
    else
        MESSAGEL("No openings for selfplay, will use empty board for opening.");

    // Set message mode to none if silence search is enabled
    if (cfg.silence)
        Config::GeneralCfg.messageMode = MsgMode::NONE;
    else
        Config::GeneralCfg.messageMode = MsgMode::BRIEF;
    Search::SearchCfg.aspirationWindow = true;

    // Set num threads and TT size
    Search::Engine.setNumThreads(cfg.numThreads);
    Search::Engine.searcher()->setMemoryLimit(cfg.hashSizeMb * 1024);

    PRNG                               prng = PRNG::nondeterministic();
    std::uniform_int_distribution<int> boardSizeDis(cfg.boardSizeMin, cfg.boardSizeMax);
    std::normal_distribution<double>   nodesDis(cfg.meanNodes, cfg.varNodes);
    size_t                             totalGamePly = 0;

    Time startTime = now(), lastTime = startTime;
    for (size_t i = 0; i < cfg.numGames;) {
        Board board(boardSizeDis(prng));
        board.newGame(cfg.rule);

        if (!cfg.silence)
            MESSAGEL("Start game " << i << ", boardsize = " << board.size()
                                   << ", rule = " << cfg.rule);

        applyOpening(board, cfg, prng);

        GameEntry gameEntry = playOneGame(board, cfg, prng, nodesDis);
        if (dataWriter)
            dataWriter->writeGame(gameEntry);

        // Print out generation progress over time
        i++;
        totalGamePly += board.ply();
        if (now() - lastTime >= cfg.reportInterval) {
            MESSAGEL("Played " << i << " of " << cfg.numGames
                               << " games, average ply = " << totalGamePly / i
                               << ", game/min = " << i / ((now() - startTime) / 60000.0));
            lastTime = now();
        }
    }

    MESSAGEL("Completed playing " << cfg.numGames << " games.");
}
