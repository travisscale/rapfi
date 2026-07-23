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

#include "featureencoder.h"

#include "../eval/scoretables.h"
#include "../game/board.h"

#include <algorithm>

namespace {

/// Packs a bit array into byte array (in big-endian).
/// @param bits The source of bits array
/// @param numBits number of bits to pack
/// @param bytes Destination of byte array
void packBitsToBytes(const uint8_t *bits, size_t numBits, uint8_t *bytes)
{
    size_t numBytesFloored = numBits / 8;
    size_t numBitsRemained = numBits % 8;

    for (size_t byteIdx = 0; byteIdx < numBytesFloored; byteIdx++) {
        *bytes++ = (bits[0] << 7) | (bits[1] << 6) | (bits[2] << 5) | (bits[3] << 4)
                   | (bits[4] << 3) | (bits[5] << 2) | (bits[6] << 1) | (bits[7] << 0);
        bits += 8;
    }

    // Deals with remaining bits that less than a byte
    *bytes = 0;
    for (size_t bitIdx = 0; bitIdx < numBitsRemained; bitIdx++)
        *bytes |= bits[bitIdx] << (7 - bitIdx);
}

}  // namespace

namespace Tuning {

void encodeEntryFeatures(const Board                               &board,
                         Pos                                        bestMove,
                         const MovePayload                         &payload,
                         Result                                     result,
                         const std::optional<std::array<float, 3>> &softValueTarget,
                         size_t                                     numCells,
                         const EntryFeatureDst                     &dst,
                         FeatureEncodeScratch                      &scratch)
{
    size_t numBytes  = (numCells + 7) / 8;
    int    boardsize = board.size();

    // Update inboard, self, oppo plane
    scratch.inBoardPlane.assign(numCells, 0);
    scratch.selfPlane.assign(numCells, 0);
    scratch.oppoPlane.assign(numCells, 0);
    Color self = board.sideToMove(), oppo = ~self;
    FOR_EVERY_POSITION(&board, pos)
    {
        int posIdx                   = pos.y() * boardsize + pos.x();
        scratch.inBoardPlane[posIdx] = true;
        scratch.selfPlane[posIdx]    = board.get(pos) == self;
        scratch.oppoPlane[posIdx]    = board.get(pos) == oppo;

        if (dst.sparseU8) {
            // Write the sparse pattern/pattern4 features
            dst.sparseU8[0 * numCells + posIdx] = board.pattern(pos, self, 0);
            dst.sparseU8[1 * numCells + posIdx] = board.pattern(pos, self, 1);
            dst.sparseU8[2 * numCells + posIdx] = board.pattern(pos, self, 2);
            dst.sparseU8[3 * numCells + posIdx] = board.pattern(pos, self, 3);
            dst.sparseU8[4 * numCells + posIdx] = board.pattern(pos, oppo, 0);
            dst.sparseU8[5 * numCells + posIdx] = board.pattern(pos, oppo, 1);
            dst.sparseU8[6 * numCells + posIdx] = board.pattern(pos, oppo, 2);
            dst.sparseU8[7 * numCells + posIdx] = board.pattern(pos, oppo, 3);
            dst.sparseU8[8 * numCells + posIdx] = board.pattern4(pos, self);
            dst.sparseU8[9 * numCells + posIdx] = board.pattern4(pos, oppo);
        }
        if (dst.sparseU16) {
            // Write the sparse pattern-code features
            dst.sparseU16[0 * numCells + posIdx] =
                self == BLACK ? board.pcode<BLACK>(pos) : board.pcode<WHITE>(pos);
            dst.sparseU16[1 * numCells + posIdx] =
                oppo == BLACK ? board.pcode<BLACK>(pos) : board.pcode<WHITE>(pos);
        }

        // Write the policy target
        dst.policyTarget[posIdx] = std::clamp<int>(
            policyTargetOf(payload, bestMove, boardsize, pos) * UINT16_MAX,
            0,
            UINT16_MAX);
    }

    // Write the policy target of the PASS move
    dst.policyTarget[numCells] = std::clamp<int>(
        policyTargetOf(payload, bestMove, boardsize, Pos::PASS) * UINT16_MAX,
        0,
        UINT16_MAX);

    // Write the packed binary planes
    packBitsToBytes(scratch.inBoardPlane.data(), numCells, dst.binaryPacked + 0 * numBytes);
    packBitsToBytes(scratch.selfPlane.data(), numCells, dst.binaryPacked + 1 * numBytes);
    packBitsToBytes(scratch.oppoPlane.data(), numCells, dst.binaryPacked + 2 * numBytes);

    // Write the global input (side to move)
    dst.globalInput[0] = (self == BLACK ? -1.0f : 1.0f);

    // Write the global value target
    if (softValueTarget.has_value()) {
        dst.globalTargets[0] = (*softValueTarget)[0];
        dst.globalTargets[1] = (*softValueTarget)[1];
        dst.globalTargets[2] = (*softValueTarget)[2];
    }
    else {
        dst.globalTargets[0] = result == RESULT_WIN;
        dst.globalTargets[1] = result == RESULT_LOSS;
        dst.globalTargets[2] = result == RESULT_DRAW;
    }
}

const std::vector<uint32_t> &sparseInputDims()
{
    static const std::vector<uint32_t> dims {
        PATTERN_NB,
        PATTERN_NB,
        PATTERN_NB,
        PATTERN_NB,
        PATTERN_NB,
        PATTERN_NB,
        PATTERN_NB,
        PATTERN_NB,
        PATTERN4_NB,
        PATTERN4_NB,
        PCODE_NB,
        PCODE_NB,
    };
    return dims;
}

}  // namespace Tuning
