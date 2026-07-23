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

#include "board.h"

#include "../config.h"

#include "../core/iohelper.h"
#include "../core/pos.h"
#include "../core/utils.h"
#include "../eval/evaluator.h"
#include "../search/searchthread.h"
#include "scopedmove.h"

#include <algorithm>
#include <cstring>  // for std::memset
#include <iomanip>
#include <sstream>
#include <tuple>

namespace {

/// Recompute the Pattern4 histogram by scanning every empty cell and check it against the
/// incrementally-maintained p4Count in the current StateInfo. Also cross-checks the empty-cell
/// bitboard against the move counters. Debug-only consistency assertion.
bool checkP4(const Board *board)
{
    // Every playable cell is either empty or holds a stone (passes occupy none), so the empty
    // bitboard's population must equal the playable cells minus the stones placed.
    if (board->emptyCells().count() != board->cellCount() - board->nonPassMoveCount())
        return false;

    int p4[SIDE_NB][PATTERN4_NB] = {0};
    FOR_EVERY_EMPTY_POS(board, pos)
    {
        p4[BLACK][board->pattern4(pos, BLACK)]++;
        p4[WHITE][board->pattern4(pos, WHITE)]++;
    }
    for (Color c : {BLACK, WHITE})
        for (Pattern4 i = FORBID; i < PATTERN4_NB; i = Pattern4(i + 1)) {
            if (p4[c][i] != board->stateInfo().p4Count[c][i])
                return false;
        }
    return true;
}

}  // namespace

Board::Board(int boardSize) : Board(boardSize, Config::GeneralCfg.defaultCandidateRange) {}

Board::Board(int boardSize, CandidateRange candRange)
    : boardSize(boardSize)
    , boardCellCount(boardSize * boardSize)
    , moveCount(0)
    , passCount {0, 0}
    , currentSide(BLACK)
    , currentZobristKey(0)
    , candidateRange(nullptr)
    , candidateRangeSize(0)
    , evaluator_(nullptr)
    , thisThread_(nullptr)
{
    assert(0 < boardSize && boardSize <= MAX_BOARD_SIZE);
    // One extra slot for the initial (pre-move) ply; *2 leaves room for pass moves on top of the
    // at-most-boardCellCount stone moves.
    stateInfos  = std::make_unique<StateInfo[]>(1 + boardCellCount * 2);
    updateCache = std::make_unique<UpdateCache[]>(1 + boardCellCount * 2);
    candJournal = std::make_unique<CandJournalEntry[]>(CAND_JOURNAL_CAPACITY);
    journalTop  = 0;

    // Set candidate range of the board. FULL_BOARD yields a null table (every empty cell is a
    // candidate, seeded once in newGame).
    CandidateRangeInfo info = candidateRangeInfo(candRange);
    candidateRange          = info.offsets;
    candidateRangeSize      = info.offsetCount;
    candAreaExpandDist      = info.expandDist;
    candStencil.build(candidateRange, candidateRangeSize);
}

Board::Board(const Board &other, Search::SearchThread *thread)
    : boardSize(other.boardSize)
    , boardCellCount(other.boardCellCount)
    , moveCount(other.moveCount)
    , passCount {other.passCount[0], other.passCount[1]}
    , currentSide(other.currentSide)
    , currentZobristKey(other.currentZobristKey)
    , candidateRange(other.candidateRange)
    , candidateRangeSize(other.candidateRangeSize)
    , candAreaExpandDist(other.candAreaExpandDist)
    , evaluator_(thread ? thread->evaluator.get() : nullptr)
    , thisThread_(thread)
{
    candStencil.build(candidateRange, candidateRangeSize);
    std::copy_n(&other.pattern2xs[0][0], FULL_BOARD_CELL_COUNT * 4, &pattern2xs[0][0]);
    std::copy_n(other.pattern4x2, FULL_BOARD_CELL_COUNT, pattern4x2);
    std::copy_n(other.bitKey0, arraySize(bitKey0), bitKey0);
    std::copy_n(other.bitKey1, arraySize(bitKey1), bitKey1);
    std::copy_n(other.bitKey2, arraySize(bitKey2), bitKey2);
    std::copy_n(other.bitKey3, arraySize(bitKey3), bitKey3);
    onBoardBB = other.onBoardBB;
    emptyBB   = other.emptyBB;

    stateInfos  = std::make_unique<StateInfo[]>(1 + boardCellCount * 2);
    updateCache = std::make_unique<UpdateCache[]>(1 + boardCellCount * 2);
    // Only copy stateinfo in [0, moveCount]
    std::copy_n(other.stateInfos.get(), 1 + moveCount, stateInfos.get());
    std::copy_n(other.updateCache.get(), 1 + moveCount, updateCache.get());

    candJournal  = std::make_unique<CandJournalEntry[]>(CAND_JOURNAL_CAPACITY);
    candidatesBB = other.candidatesBB;
    journalTop   = other.journalTop;
    std::copy_n(other.candJournal.get(), other.journalTop, candJournal.get());

    // Sync evaluator state with board state
    if (evaluator_)
        evaluator_->syncWithBoard(*this);
}

template <Rule R>
void Board::newGame()
{
    // Reset to an empty board, then compute every empty cell's patterns from scratch once;
    // all later positions are reached incrementally through move()/undo(). Wall cells' array
    // entries stay zero forever — required for the pre-legality frozen reads.
    std::memset(pattern2xs, 0, sizeof(pattern2xs));
    std::memset(pattern4x2, 0, sizeof(pattern4x2));
    std::fill_n(bitKey0, arraySize(bitKey0), 0);
    std::fill_n(bitKey1, arraySize(bitKey1), 0);
    std::fill_n(bitKey2, arraySize(bitKey2), 0);
    std::fill_n(bitKey3, arraySize(bitKey3), 0);
    onBoardBB.zero();
    emptyBB.zero();
    candidatesBB.zero();
    journalTop = 0;

    // Init board state to empty
    moveCount         = 0;
    passCount[BLACK]  = 0;
    passCount[WHITE]  = 0;
    currentSide       = BLACK;
    currentZobristKey = Hash::zobrist[BLACK][FULL_BOARD_CELL_COUNT - 1];
    for (Pos i = Pos::FULL_BOARD_START; i < Pos::FULL_BOARD_END; i++) {
        if (!i.isInBoard(boardSize, boardSize))
            continue;

        // Seed empty cells with both color bits set (encoding 11); walls keep 00. Placing a stone
        // later toggles one bit, leaving the opposite color's bit set (black 10, white 01).
        setBitKey(i, BLACK);
        setBitKey(i, WHITE);
        onBoardBB.set(i);
        emptyBB.set(i);
    }

    // Init state info of the first ply with rule R (the memset zeroes journalStart too,
    // matching journalTop == 0 above)
    StateInfo &st = stateInfos[moveCount];
    std::memset(&st, 0, sizeof(StateInfo));

    Value valueBlack = VALUE_ZERO;
    FOR_EVERY_POSITION(this, pos)
    {
        Pattern2x *p2x = pattern2xs[pos];
        for (int dir = 0; dir < 4; dir++) {
            p2x[dir] = PatternConfig::lookupPattern<R>(getKeyAt<R>(pos, dir));
            assert(p2x[dir].patBlack <= F1 && p2x[dir].patWhite <= F1);
        }

        PatternCode pcodeBlack = pcode<BLACK>(pos), pcodeWhite = pcode<WHITE>(pos);
        Pattern4    p4Black    = (Pattern4)Evaluation::getP4Score(R, BLACK, pcodeBlack);
        Pattern4    p4White    = (Pattern4)Evaluation::getP4Score(R, WHITE, pcodeWhite);
        pattern4x2[pos]        = uint8_t(p4Black) | uint8_t(p4White) << 4;
        st.p4Count[BLACK][p4Black]++;
        st.p4Count[WHITE][p4White]++;
        valueBlack += Evaluation::getValueBlack(R, pcodeBlack, pcodeWhite);
    }
    st.valueBlack = valueBlack;
    st.candArea   = CandArea();

    // For full board candidate range, we manually set all empty cells to candidates.
    if (candidateRangeSize == 0)
        expandCandArea(centerPos(), size() / 2, 0);

    assert(checkP4(this));

    // Reset evaluator state to empty board
    if (evaluator_)
        evaluator_->initEmptyBoard();
}

template void Board::newGame<FREESTYLE>();
template void Board::newGame<STANDARD>();
template void Board::newGame<RENJU>();

template <Rule R, Board::MoveType MT>
void Board::move(Pos pos)
{
    // A pass copies the previous ply unchanged apart from the side to move and pass counter.
    if (UNLIKELY(pos == Pos::PASS)) {
        assert(passMoveCount() < cellCount());

        StateInfo &st   = stateInfos[++moveCount];
        st              = stateInfos[moveCount - 1];
        st.lastMove     = Pos::PASS;
        st.journalStart = journalTop;  // overwrite the copied anchor: this ply owns new entries

        passCount[currentSide]++;
        currentSide = ~currentSide;

        // after move evaluator update
        if (MT == MoveType::NORMAL && evaluator_)
            evaluator_->afterPass(*this);
        return;
    }

    assert(pos.valid());
    assert(isEmpty(pos));

    // before move evaluator update
    if (MT == MoveType::NORMAL && evaluator_)
        evaluator_->beforeMove(*this, pos);

    UpdateCache &pc = updateCache[moveCount];
    StateInfo   &st = stateInfos[++moveCount];
    st              = stateInfos[moveCount - 1];
    st.lastMove     = pos;
    st.journalStart = journalTop;  // overwrite the copied anchor: this ply owns new entries
    st.candArea.expand(pos, boardSize, candAreaExpandDist);

    currentZobristKey ^= Hash::zobrist[currentSide][pos];
    flipBitKey(pos, currentSide);
    emptyBB.clear(pos);

    Value deltaValueBlack            = VALUE_ZERO;
    int   f4CountBeforeMove[SIDE_NB] = {p4Count(BLACK, B_FLEX4), p4Count(WHITE, B_FLEX4)};
    int   updateCacheIdx             = 0;

    // Placing a stone only changes the line patterns of the cells within L steps of `pos` along
    // each of the 4 directions. We walk those cells from i=-L to i=+L (skipping i=0, the stone
    // itself), and instead of recomputing a rotated line key per cell we keep `bitKey[dir]`
    // pre-aligned to the i=-L cell and shift it one cell (2 bits) per step. Stepping past the
    // center (i=-1 -> i=+1) skips two cells, hence the doubled increment and shift amount below.
    constexpr int L         = PatternConfig::HalfLineLen<R>;
    int           x         = pos.x() + BOARD_BOUNDARY;
    int           y         = pos.y() + BOARD_BOUNDARY;
    uint64_t      bitKey[4] = {
        rotr(bitKey0[y], 2 * (x - 2 * L)),
        rotr(bitKey1[x], 2 * (y - 2 * L)),
        rotr(bitKey2[x + y], 2 * (x - 2 * L)),
        rotr(bitKey3[FULL_BOARD_SIZE - 1 - x + y], 2 * (x - 2 * L)),
    };

    for (int i = -L; i <= L; i += 1 + (i == -1)) {
        for (int dir = 0; dir < 4; dir++) {
            Pos posi = pos + DIRECTION[dir] * i;
            // Current cell's 2-bit occupancy code sits at bit offset 2L of the pre-rotated
            // window; 0b11 = empty. Replaces the per-cell piece load.
            // Debug cross-check (keep through stage 6): register test must agree with emptyBB.
            assert((((bitKey[dir] >> (2 * L)) & 0b11) == 0b11) == emptyBB.test(posi));
            if (((bitKey[dir] >> (2 * L)) & 0b11) != 0b11)
                continue;

            Pattern2x *p2x    = pattern2xs[posi];
            uint8_t    p4Pair = pattern4x2[posi];

            // Save the cell's pre-update line pattern and pattern4 pair so undo() can
            // restore them verbatim (captured before any write to the cell).
            pc[updateCacheIdx].pattern2x    = p2x[dir];
            pc[updateCacheIdx].pattern4Pair = p4Pair;
            updateCacheIdx++;

            // PCODE is order-independent, so index it with the changed direction last:
            // the three unchanged directions give one base row per color, and the old and
            // new pcode are two loads from that same 16-entry row.
            const int          d1 = (dir + 1) & 3, d2 = (dir + 2) & 3, d3 = (dir + 3) & 3;
            const PatternCode *rowBlack =
                PatternConfig::PCODE[p2x[d1].patBlack][p2x[d2].patBlack][p2x[d3].patBlack];
            const PatternCode *rowWhite =
                PatternConfig::PCODE[p2x[d1].patWhite][p2x[d2].patWhite][p2x[d3].patWhite];
            const Pattern oldBlack = p2x[dir].patBlack;
            const Pattern oldWhite = p2x[dir].patWhite;

            p2x[dir] = PatternConfig::lookupPattern<R>(bitKey[dir]);

            PatternCode pcodeBlack = rowBlack[p2x[dir].patBlack];
            PatternCode pcodeWhite = rowWhite[p2x[dir].patWhite];

            if constexpr (MT == MoveType::NORMAL || MT == MoveType::NO_EVALUATOR) {
                // The old value is no longer cached on the cell: rebuild it from the old
                // pcodes (same-row loads) so the incremental sum stays bit-identical.
                deltaValueBlack +=
                    Evaluation::getValueBlack(R, pcodeBlack, pcodeWhite)
                    - Evaluation::getValueBlack(R, rowBlack[oldBlack], rowWhite[oldWhite]);
            }

            Pattern4 p4Black = (Pattern4)Evaluation::getP4Score(R, BLACK, pcodeBlack);
            Pattern4 p4White = (Pattern4)Evaluation::getP4Score(R, WHITE, pcodeWhite);
            st.p4Count[BLACK][p4Pair & 0xF]--;
            st.p4Count[WHITE][p4Pair >> 4]--;
            st.p4Count[BLACK][p4Black]++;
            st.p4Count[WHITE][p4White]++;
            pattern4x2[posi] = uint8_t(p4Black) | uint8_t(p4White) << 4;

            if (p4Black >= C_BLOCK4_FLEX3)
                st.lastPattern4Move[BLACK][p4Black - C_BLOCK4_FLEX3] = posi;
            if (p4White >= C_BLOCK4_FLEX3)
                st.lastPattern4Move[WHITE][p4White - C_BLOCK4_FLEX3] = posi;
        }

        const int shamt = 2 + 2 * (i == -1);
        bitKey[0] >>= shamt;
        bitKey[1] >>= shamt;
        bitKey[2] >>= shamt;
        bitKey[3] >>= shamt;
    }

    // The placed cell is no longer empty: drop its own (now stale) value and p4 contributions
    // that were folded in above. Its pattern2x entries are frozen at their valid empty-cell
    // state (the walk skips i == 0), so the stale value is rebuilt from its pcodes.
    if constexpr (MT == MoveType::NORMAL || MT == MoveType::NO_EVALUATOR) {
        st.valueBlack += deltaValueBlack
                         - Evaluation::getValueBlack(R, pcode<BLACK>(pos), pcode<WHITE>(pos));
    }
    st.p4Count[BLACK][pattern4x2[pos] & 0xF]--;
    st.p4Count[WHITE][pattern4x2[pos] >> 4]--;

    if (MT != MoveType::NO_EVAL_MULTI)
        currentSide = ~currentSide;

    assert(checkP4(this));
    assert(updateCacheIdx <= std::tuple_size_v<UpdateCache>);

    // Mark the new stone's candidate neighborhood in the board-level candidate bitboard,
    // journaling the 0->1 transitions so undo() can rewind them.
    Bitboard::forEachStencilWord(candStencil, pos, [this](int w, uint64_t bits) {
        setCandidateBits(w, bits);
    });

    // If this move is what first created a flex four for a side, remember it as that side's
    // flex-four attack move (used by the four-defence move generator).
    for (Color c : {BLACK, WHITE}) {
        if (!f4CountBeforeMove[c] && p4Count(c, B_FLEX4))
            st.lastFlex4AttackMove[c] = pos;
    }

    // after move evaluator update
    if (MT == MoveType::NORMAL && evaluator_)
        evaluator_->afterMove(*this, pos);
}

template void Board::move<FREESTYLE, Board::MoveType::NORMAL>(Pos pos);
template void Board::move<FREESTYLE, Board::MoveType::NO_EVAL>(Pos pos);
template void Board::move<STANDARD, Board::MoveType::NORMAL>(Pos pos);
template void Board::move<STANDARD, Board::MoveType::NO_EVAL>(Pos pos);
template void Board::move<RENJU, Board::MoveType::NORMAL>(Pos pos);
template void Board::move<RENJU, Board::MoveType::NO_EVAL>(Pos pos);

template <Rule R, Board::MoveType MT>
void Board::undo()
{
    assert(moveCount > 0);
    Pos lastPos = getLastMove();

    // Rewind this ply's candidate-bit transitions (XOR clears exactly the bits that went
    // 0->1 at this ply; disjointness makes order irrelevant). Passes need this too:
    // expandCandArea may have appended entries on a pass ply.
    {
        const StateInfo &dying = stateInfos[moveCount];
        for (uint32_t j = journalTop; j-- > dying.journalStart;)
            candidatesBB.words[candJournal[j].wordIdx] ^= candJournal[j].delta;
        journalTop = dying.journalStart;
    }

    // Undoing a pass just restores the side to move and pass counter.
    if (UNLIKELY(lastPos == Pos::PASS)) {
        currentSide = ~currentSide;
        assert(passCount[currentSide] > 0);
        passCount[currentSide]--;
        moveCount--;

        // after undo evaluator update
        if (MT == MoveType::NORMAL && evaluator_)
            evaluator_->afterUndoPass(*this);
        return;
    }

    // before undo evaluator update
    if (MT == MoveType::NORMAL && evaluator_)
        evaluator_->beforeUndo(*this, lastPos);

    if (MT != MoveType::NO_EVAL_MULTI)
        currentSide = ~currentSide;
    assert(get(lastPos) == currentSide);

    flipBitKey(lastPos, currentSide);
    currentZobristKey ^= Hash::zobrist[currentSide][lastPos];
    emptyBB.set(lastPos);

    moveCount--;
    const UpdateCache &pc             = updateCache[moveCount];
    int                updateCacheIdx = 0;

    // Mirror move()'s line walk over the same cells in the same order, restoring each cell's
    // changed-direction line pattern and pattern4 pair straight from the saved UpdateCache —
    // no bitkey window or table lookup needed. Occupancy is unchanged between move() and here
    // (the stone was removed above), so emptyBB selects exactly the same cell set.
    constexpr int L = PatternConfig::HalfLineLen<R>;
    for (int i = -L; i <= L; i += 1 + (i == -1)) {
        for (int dir = 0; dir < 4; dir++) {
            Pos posi = lastPos + DIRECTION[dir] * i;
            if (!emptyBB.test(posi))
                continue;

            pattern2xs[posi][dir] = pc[updateCacheIdx].pattern2x;
            pattern4x2[posi]      = pc[updateCacheIdx].pattern4Pair;
            updateCacheIdx++;
        }
    }

    assert(checkP4(this));
    assert(updateCacheIdx <= std::tuple_size_v<UpdateCache>);
    // Journal round-trip invariant: the rewind above restored the dying ply's anchor.
    assert(journalTop == stateInfos[moveCount + 1].journalStart);

    // after undo evaluator update
    if (MT == MoveType::NORMAL && evaluator_)
        evaluator_->afterUndo(*this, lastPos);
}

template void Board::undo<FREESTYLE, Board::MoveType::NORMAL>();
template void Board::undo<FREESTYLE, Board::MoveType::NO_EVAL>();
template void Board::undo<STANDARD, Board::MoveType::NORMAL>();
template void Board::undo<STANDARD, Board::MoveType::NO_EVAL>();
template void Board::undo<RENJU, Board::MoveType::NORMAL>();
template void Board::undo<RENJU, Board::MoveType::NO_EVAL>();

bool Board::checkForbiddenPoint(Pos pos) const
{
    // Renju forbids black from playing an overline, a double-four, or a double-three. The pattern
    // tables flag candidates as FORBID, but some are false positives (e.g. a "double three" where
    // one three can only be completed via another forbidden point is not actually forbidden), so
    // overline and double-four are confirmed directly from the line patterns while double-three is
    // verified by placing the stone and counting threes that lead to a genuine win.
    if (pattern4(pos, BLACK) != FORBID)
        return false;

    int winByFour = 0;
    for (int dir = 0; dir < 4; dir++) {
        Pattern p = pattern(pos, BLACK, dir);
        // If this forbidden point is a Overline, it must be a true forbidden point.
        if (p == OL)
            return true;
        // Otherwise if it has at least two Four(B4/F4), it must be a true forbidden point.
        else if (p == B4 || p == B4S || p == F4) {
            if (++winByFour >= 2)
                return true;
        }
    }

    // Check the remaining false-forbidden cases by placing black at pos and recursing. The guards
    // restore the side to move and undo the stone on every exit path of this function. Reads of
    // the probe cell after the ScopedMove return its frozen placement-time patterns (the freeze
    // invariant), which is exactly the pre-placement state this check needs.
    ScopedSwitchSide                                        asBlack(*this, BLACK);
    ScopedMove<Rule::RENJU, Board::MoveType::NO_EVAL_MULTI> probe(*this, pos);

    constexpr int MaxFindDist = 4;
    int           winByThree  = 0;

    for (int dir = 0; dir < 4; dir++) {
        // Only look line that is possible to become a FLEX4 or FIVE
        Pattern p = pattern(pos, BLACK, dir);

        // double three forbidden type
        if (p != F3 && p != F3S)  // p must be one of F3, F3S
            continue;

        Pos posi = pos;
        for (int i = 0; i < MaxFindDist; i++) {
            posi -= DIRECTION[dir];

            if (Color piece = get(posi); piece == EMPTY) {
                if (pattern4(posi, BLACK) == B_FLEX4 || pattern(posi, BLACK, dir) == F5
                    || pattern4(posi, BLACK) == FORBID && pattern(posi, BLACK, dir) == F4
                           && !checkForbiddenPoint(posi)) {
                    winByThree++;
                    goto next_direction;
                }
                break;
            }
            else if (piece != BLACK)
                break;
        }
        posi = pos;
        for (int i = 0; i < MaxFindDist; i++) {
            posi += DIRECTION[dir];

            if (Color piece = get(posi); piece == EMPTY) {
                if (pattern4(posi, BLACK) == B_FLEX4 || pattern(posi, BLACK, dir) == F5
                    || pattern4(posi, BLACK) == FORBID && pattern(posi, BLACK, dir) == F4
                           && !checkForbiddenPoint(posi)) {
                    winByThree++;
                    goto next_direction;
                }
                break;
            }
            else if (piece != BLACK)
                break;
        }

    next_direction:
        if (winByThree >= 2)
            break;
    }

    return winByThree >= 2;
}

Score Board::score(Rule rule, Pos pos, Color side) const
{
    Pattern4Score p4ScoreBlack = Evaluation::getP4Score(rule, BLACK, pcode<BLACK>(pos));
    Pattern4Score p4ScoreWhite = Evaluation::getP4Score(rule, WHITE, pcode<WHITE>(pos));
    return side == BLACK ? p4ScoreBlack.scoreSelf() + p4ScoreWhite.scoreOppo()
                         : p4ScoreWhite.scoreSelf() + p4ScoreBlack.scoreOppo();
}

Pos Board::getLastActualMoveOfSide(Color side) const
{
    assert(side == BLACK || side == WHITE);

    for (int reverseIdx = 0; reverseIdx < moveCount; reverseIdx++) {
        Pos move = getRecentMove(reverseIdx);
        if (move == Pos::PASS)
            continue;
        if (get(move) == side)
            return move;
    }

    return Pos::NONE;
}

void Board::expandCandArea(Pos pos, int fillDist, int lineDist)
{
    StateInfo &st = stateInfos[moveCount];
    int        x = pos.x(), y = pos.y();

    // Grow the bounding box too: FOR_EVERY_CAND_POS masks iteration by candArea, so candidate
    // bits set outside the box would never be visited.
    st.candArea.expand(pos, boardSize, std::max(fillDist, lineDist));

    auto markCandidate = [&](Pos p) {
        if (p >= 0 && p < FULL_BOARD_CELL_COUNT && isEmpty(p))
            setCandidateBits(p >> 6, uint64_t(1) << (p & 63));
    };

    for (int i = std::max(3, fillDist + 1); i <= lineDist; i++) {
        for (int dir = 0; dir < 4; dir++)
            markCandidate(pos + DIRECTION[dir] * i);
    }
    for (int xi = -fillDist; xi <= fillDist; xi++) {
        for (int yi = -fillDist; yi <= fillDist; yi++)
            markCandidate(Pos {x + xi, y + yi});
    }
}

std::string Board::positionString() const
{
    std::stringstream ss;
    for (int i = 0; i < ply(); i++) {
        Pos pos = getHistoryMove(i);
        if (pos == Pos::PASS)
            ss << "--";
        else
            ss << char('a' + pos.x()) << (1 + pos.y());
    }
    return ss.str();
}

std::string Board::trace(Rule rule) const
{
    std::stringstream ss;
    const StateInfo  &st = stateInfo();

    ss << "Hash: " << std::hex << zobristKey() << std::dec << '\n';
    ss << "Ply: " << ply() << "\n";
    ss << "NonPassCount: " << nonPassMoveCount() << "\n";
    ss << "PassCount[Black]: " << passMoveCountOfSide(BLACK)
       << "  PassCount[White]: " << passMoveCountOfSide(WHITE) << "\n";
    ss << "SideToMove: " << sideToMove() << "\n";
    ss << "LastPos: " << getLastMove() << '\n';
    ss << "Eval[Black]: " << st.valueBlack << '\n';
    ss << "LastP4[Black][A]: " << st.lastPattern4(BLACK, A_FIVE)
       << "  LastP4[White][A]: " << st.lastPattern4(WHITE, A_FIVE) << '\n';
    ss << "LastP4[Black][B]: " << st.lastPattern4(BLACK, B_FLEX4)
       << "  LastP4[White][B]: " << st.lastPattern4(WHITE, B_FLEX4) << '\n';
    ss << "LastP4[Black][C]: " << st.lastPattern4(BLACK, C_BLOCK4_FLEX3)
       << "  LastP4[White][C]: " << st.lastPattern4(WHITE, C_BLOCK4_FLEX3) << '\n';
    ss << "LastF4[Black]: " << st.lastFlex4AttackMove[BLACK]
       << "  LastF4[White]: " << st.lastFlex4AttackMove[WHITE] << '\n';

    auto printBoard = [&](auto &&posTextFunc, int textWidth = 1) {
        FOR_EVERY_POSITION(this, pos)
        {
            int x = pos.x(), y = pos.y();
            if (x != 0 || y != 0)
                ss << ' ';
            if (x == 0 && y != 0)
                ss << '\n';
            posTextFunc(pos);
            if (x == size() - 1)
                ss << ' ' << y + 1;
        }
        ss << '\n';
        for (int x = 0; x < size(); x++)
            ss << std::setw(textWidth) << char(x + 65) << " ";
        ss << '\n';
    };

    auto printPiece = [&](Pos pos) {
        switch (get(pos)) {
        case BLACK: ss << 'X'; break;
        case WHITE: ss << 'O'; break;
        case EMPTY: ss << (isCandidate(pos) ? '*' : '.'); break;
        default: ss << ' '; break;
        }
    };

    ss << "----------------Board----------------\n";
    printBoard(printPiece);

    ss << "----------Pattern4----Black----------\n";
    printBoard([&](Pos pos) {
        if (isEmpty(pos))
            ss << pattern4(pos, BLACK);
        else
            ss << '.';
    });

    ss << "----------Pattern4----White----------\n";
    printBoard([&](Pos pos) {
        if (isEmpty(pos))
            ss << pattern4(pos, WHITE);
        else
            ss << '.';
    });

    ss << "----------Score-------Black----------\n";
    printBoard(
        [&](Pos pos) {
            if (isEmpty(pos))
                ss << std::setw(3) << score(rule, pos, BLACK);
            else {
                ss << '[';
                printPiece(pos);
                ss << ']';
            }
        },
        3);

    ss << "----------Score-------White----------\n";
    printBoard(
        [&](Pos pos) {
            if (isEmpty(pos))
                ss << std::setw(3) << score(rule, pos, WHITE);
            else {
                ss << '[';
                printPiece(pos);
                ss << ']';
            }
        },
        3);

    if (evaluator_) {
        Evaluation::PolicyBuffer policyBuf(boardSize);
        policyBuf.setComputeFlagForAllEmptyCell(*this);

        // Calculate policy for the current side
        evaluator_->evaluatePolicy(*this, policyBuf);

        ss << "----------Policy------Self-----------\n";
        printBoard(
            [&](Pos pos) {
                if (isEmpty(pos))
                    ss << std::setw(4) << policyBuf.score(pos);
                else {
                    ss << " [";
                    printPiece(pos);
                    ss << "]";
                }
            },
            4);

        // Calculate for the opponent side
        {
            ScopedSwitchSide oppoSide(*this, ~sideToMove());
            evaluator_->evaluatePolicy(*this, policyBuf);
        }

        ss << "----------Policy------Oppo-----------\n";
        printBoard(
            [&](Pos pos) {
                if (isEmpty(pos))
                    ss << std::setw(4) << policyBuf.score(pos);
                else {
                    ss << " [";
                    printPiece(pos);
                    ss << "]";
                }
            },
            4);
    }

    return ss.str();
}
