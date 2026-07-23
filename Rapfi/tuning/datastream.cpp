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

#include "datastream.h"

#include "../core/compressor.h"

#include <cassert>
#include <stdexcept>

namespace Tuning {

MultiFileInputStream::MultiFileInputStream(const std::vector<std::string> &filenames)
    : nextFileIdx_(0)
    , istream_(nullptr)
{
    if (filenames.empty())
        throw std::runtime_error("no file in dataset");

    files_.reserve(filenames.size());
    for (const std::string &filename : filenames) {
        std::ifstream fileStream(filename, std::ios::binary);
        if (!fileStream.is_open())
            throw std::runtime_error("unable to open file " + filename);

        fileStream.exceptions(std::istream::badbit | std::istream::failbit);
        files_.push_back(std::move(fileStream));
    }

    nextFile();
}

// Out-of-line so the header can hold Compressor by unique_ptr as an incomplete type.
MultiFileInputStream::~MultiFileInputStream() = default;

bool MultiFileInputStream::nextFile()
{
    if (nextFileIdx_ == files_.size())
        return false;

    // Delete the previous compressor (if any) before rebinding to the next file
    if (compressor_) {
        istream_ = nullptr;
        compressor_.reset();
    }

    // Probe the LZ4 frame magic. A zero-byte file skips the probe (the read would
    // throw, as the streams have failbit exceptions enabled; peek only sets eofbit)
    // and is treated as an uncompressed stream, which yields EOF on the first read
    // and gets skipped by the caller's end-of-file advance loop.
    int magic = 0;
    if (files_[nextFileIdx_].peek() != std::ifstream::traits_type::eof()) {
        files_[nextFileIdx_].read(reinterpret_cast<char *>(&magic), sizeof(magic));
        files_[nextFileIdx_].seekg(0);
    }

    compressor_ = std::make_unique<Compressor>(
        files_[nextFileIdx_],
        magic == 0x184D2204 ? Compressor::Type::LZ4_DEFAULT : Compressor::Type::NO_COMPRESS);
    istream_ = compressor_->openInputStream();
    nextFileIdx_++;

    if (!istream_)
        throw std::runtime_error("unable to open dataset stream");

    return true;
}

void MultiFileInputStream::reset()
{
    istream_ = nullptr;
    if (compressor_)
        compressor_.reset();
    for (std::ifstream &fs : files_) {
        fs.clear();  // drop any eof/fail state before rewinding
        fs.seekg(0);
    }
    nextFileIdx_ = 0;
    nextFile();
}

}  // namespace Tuning
