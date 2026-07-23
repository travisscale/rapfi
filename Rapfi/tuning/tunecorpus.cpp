/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#include "tunecorpus.h"

#include <iterator>
#include <stdexcept>
#include <string>

namespace Tuning {
namespace {

uint32_t checkedOffset(size_t current, size_t added, const char *section)
{
    constexpr size_t MaxOffset = std::numeric_limits<uint32_t>::max();
    if (current > MaxOffset || added > MaxOffset - current)
        throw std::length_error(std::string("prepared corpus ") + section
                                + " exceeds 32-bit shard offsets");
    return static_cast<uint32_t>(current + added);
}

template <typename T>
size_t vectorCapacityBytes(const std::vector<T> &values)
{
    return values.capacity() * sizeof(T);
}

}  // namespace

PreparedCorpus::PreparedCorpus() : evalOffsets_ {0}, policyOffsets_ {0} {}

void PreparedCorpus::clear()
{
    results_.clear();
    staticEvals_.clear();
    bestCandidates_.clear();
    evalTerms_.clear();
    policyCandidates_.clear();
    evalOffsets_.assign(1, 0);
    policyOffsets_.assign(1, 0);
}

void PreparedCorpus::reserveSamples(size_t count)
{
    results_.reserve(count);
    staticEvals_.reserve(count);
    bestCandidates_.reserve(count);
    evalOffsets_.reserve(count + 1);
    policyOffsets_.reserve(count + 1);
}

void PreparedCorpus::append(uint8_t                             resultTimesTwo,
                            int16_t                             staticEval,
                            const std::vector<TuneCoeff>       &evalTerms,
                            const std::vector<PolicyCandidate> &policyCandidates,
                            uint16_t                            bestCandidate)
{
    if (resultTimesTwo > 2)
        throw std::invalid_argument("prepared sample result is out of range");
    if (bestCandidate != NoPolicyTarget && bestCandidate >= policyCandidates.size())
        throw std::invalid_argument("prepared sample policy target is out of range");

    uint32_t evalEnd = checkedOffset(evalTerms_.size(), evalTerms.size(), "value terms");
    uint32_t policyEnd =
        checkedOffset(policyCandidates_.size(), policyCandidates.size(), "policy candidates");

    evalTerms_.insert(evalTerms_.end(), evalTerms.begin(), evalTerms.end());
    policyCandidates_.insert(
        policyCandidates_.end(), policyCandidates.begin(), policyCandidates.end());
    results_.push_back(resultTimesTwo);
    staticEvals_.push_back(staticEval);
    bestCandidates_.push_back(bestCandidate);
    evalOffsets_.push_back(evalEnd);
    policyOffsets_.push_back(policyEnd);
}

void PreparedCorpus::append(PreparedCorpus &&other)
{
    if (other.empty())
        return;

    checkedOffset(evalTerms_.size(), other.evalTerms_.size(), "value terms");
    checkedOffset(
        policyCandidates_.size(), other.policyCandidates_.size(), "policy candidates");

    // Take over the first complete fragment without copying. Later inserts
    // deliberately rely on vector's geometric growth instead of reserving the
    // exact new total for every fragment, which would repeatedly copy the
    // complete accumulated corpus.
    if (empty()) {
        *this = std::move(other);
        return;
    }

    uint32_t evalBase   = static_cast<uint32_t>(evalTerms_.size());
    uint32_t policyBase = static_cast<uint32_t>(policyCandidates_.size());

    evalTerms_.insert(evalTerms_.end(),
                      std::make_move_iterator(other.evalTerms_.begin()),
                      std::make_move_iterator(other.evalTerms_.end()));
    policyCandidates_.insert(policyCandidates_.end(),
                             std::make_move_iterator(other.policyCandidates_.begin()),
                             std::make_move_iterator(other.policyCandidates_.end()));
    results_.insert(results_.end(), other.results_.begin(), other.results_.end());
    staticEvals_.insert(staticEvals_.end(), other.staticEvals_.begin(), other.staticEvals_.end());
    bestCandidates_.insert(
        bestCandidates_.end(), other.bestCandidates_.begin(), other.bestCandidates_.end());

    for (size_t i = 1; i < other.evalOffsets_.size(); i++)
        evalOffsets_.push_back(evalBase + other.evalOffsets_[i]);
    for (size_t i = 1; i < other.policyOffsets_.size(); i++)
        policyOffsets_.push_back(policyBase + other.policyOffsets_[i]);
}

size_t PreparedCorpus::capacityBytes() const
{
    return vectorCapacityBytes(results_) + vectorCapacityBytes(staticEvals_)
           + vectorCapacityBytes(bestCandidates_) + vectorCapacityBytes(evalOffsets_)
           + vectorCapacityBytes(evalTerms_) + vectorCapacityBytes(policyOffsets_)
           + vectorCapacityBytes(policyCandidates_);
}

}  // namespace Tuning
