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

#pragma once

// The classical eval tables and their query API live in namespace Evaluation in
// eval/scoretables.h. Hot-path headers should include that header directly instead
// of this one.
#include "eval/scoretables.h"

#include "core/pos.h"
#include "core/types.h"

#include <cstddef>
#include <memory>
#include <string>

/// MsgMode represents the message mode that controls how messages are outputed in search.
enum class MsgMode { NONE, BRIEF, NORMAL, UCILIKE };

namespace Search {
class Searcher;  // forward declaration
}

namespace Database {
class DBStorage;           // forward declaration
enum class OverwriteRule;  // forward declaration
}  // namespace Database

namespace Config {

extern const std::string InternalConfig;

// -------------------------------------------------
// Model configs
// (ScalingFactor and the EVALS/EVALS_THREAT/P4SCORES tables are declared in
// eval/scoretables.h together with their query API)
extern float EvaluatorMarginWinLossScale;
extern float EvaluatorMarginWinLossExponent;
extern float EvaluatorMarginScale;
extern float EvaluatorDrawBlackWinRate;
extern float EvaluatorDrawRatio;

// -------------------------------------------------
// General options
extern bool                ReloadConfigEachMove;
extern bool                ClearHashAfterConfigLoaded;
extern size_t              DefaultThreadNum;
extern MsgMode             MessageMode;
extern CoordConvertionMode IOCoordMode;
extern CandidateRange      DefaultCandidateRange;
extern size_t              MemoryReservedMB[RULE_NB];
extern size_t              DefaultTTSizeKB;

// -------------------------------------------------
// Search options
extern bool AspirationWindow;
extern bool FilterSymmetryRootMoves;
extern int  NumIterationAfterMate;
extern int  NumIterationAfterSingularRoot;
extern int  MaxSearchDepth;

extern bool  ExpandWhenFirstEvaluate;
extern int   MaxNumVisitsPerPlayout;
extern int   NodesToPrintMCTSRootmoves;
extern int   TimeToPrintMCTSRootmoves;
extern int   MaxNonPVRootmovesToPrint;
extern int   NumNodesAfterSingularRoot;
extern int   NumNodeTableShardsPowerOfTwo;
extern float DrawUtilityPenalty;

// -------------------------------------------------
// Time management options
extern int   TurnTimeReserved;
extern float MatchSpace;
extern float MatchSpaceMin;
extern float AverageBranchFactor;
extern float AdvancedStopRatio;
extern int   MoveHorizon;
extern float TimeDivisorBias;
extern float TimeDivisorScale;
extern float TimeDivisorDepthPow;
extern float FallingFactorScale;
extern float FallingFactorBias;
extern float BestmoveStableReductionScale;
extern float BestmoveStablePrevReductionPow;

// -------------------------------------------------
// Database options
extern bool        DatabaseDefaultEnabled;
extern uint16_t    DatabaseLegacyFileCodePage;
extern std::string DatabaseType;
extern std::string DatabaseURL;
extern size_t      DatabaseCacheSize;
extern size_t      DatabaseRecordCacheSize;

// Library import options
extern char DatabaseLibBlackWinMark;
extern char DatabaseLibWhiteWinMark;
extern char DatabaseLibBlackLoseMark;
extern char DatabaseLibWhiteLoseMark;
extern bool DatabaseLibIgnoreComment;
extern bool DatabaseLibIgnoreBoardText;

// Database search options
extern bool                      DatabaseReadonlyMode;
extern bool                      DatabaseMandatoryParentWrite;
extern int                       DatabaseQueryPly;
extern int                       DatabaseQueryPVIterPerPlyIncrement;
extern int                       DatabaseQueryNonPVIterPerPlyIncrement;
extern int                       DatabasePVWritePly;
extern int                       DatabasePVWriteMinDepth;
extern int                       DatabaseNonPVWritePly;
extern int                       DatabaseNonPVWriteMinDepth;
extern int                       DatabaseWriteValueRange;
extern int                       DatabaseMateWritePly;
extern int                       DatabaseMateWriteMinDepthExact;
extern int                       DatabaseMateWriteMinDepthNonExact;
extern int                       DatabaseMateWriteMinStep;
extern int                       DatabaseExactOverwritePly;
extern int                       DatabaseNonExactOverwritePly;
extern ::Database::OverwriteRule DatabaseOverwriteRule;
extern int                       DatabaseOverwriteExactBias;
extern int                       DatabaseOverwriteDepthBoundBias;
extern int                       DatabaseQueryResultDepthBoundBias;

// -------------------------------------------------
// Config loading & exporting

/// Loads TOML config from a text stream.
/// @param configStream Input config text stream.
/// @param skipModelLoading Whether to skip model loading. This can be set when
/// model is overrided in command line.
/// @return True if config is loaded successfully.
bool loadConfig(std::istream &configStream);

/// Load a LZ4 compressed classical evaluation model from a binary stream.
/// @return Returns true if loaded successfully, otherwise returns false.
bool loadModel(std::istream &inStream);

/// Exports current classic evaluation model to a binary stream.
void exportModel(std::ostream &outStream);

// -------------------------------------------------
// Searcher loading

/// Create a searcher instance from config.
/// @param searcherName The name of the searcher. Empty string to use default.
/// @return The popinter to the searcher. Can not be nullptr.
std::unique_ptr<::Search::Searcher> createSearcher(std::string searcherName = "");

// -------------------------------------------------
// Database loading

/// Create a default database storage instance from config.
/// @param url URL of the database, empty for default url from config.
/// @return The pointer to DBStorage instance, or nullptr if can not create.
std::unique_ptr<::Database::DBStorage> createDefaultDBStorage(std::string url = "");

}  // namespace Config
