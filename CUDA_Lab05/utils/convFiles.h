#pragma once

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstddef>

namespace conv
{

    enum class FileStatus : int
    {
        Success = 0,
        NoFile = 1
    };

    // --- 1D ---
    inline FileStatus getSizes1D(const std::string &fileName, int &maskSize, int &signalSize) noexcept
    {
        std::ifstream in(fileName);
        if (!in)
            return FileStatus::NoFile;

        in >> maskSize >> signalSize;
        return FileStatus::Success;
    }

    inline FileStatus getValues1D(const std::string &fileName, std::vector<float> &maskValues, std::vector<float> &signalValues)
    {
        std::ifstream in(fileName);
        if (!in)
            return FileStatus::NoFile;

        int maskSize = 0, signalSize = 0;
        in >> maskSize >> signalSize;

        maskValues.resize(static_cast<std::size_t>(maskSize));
        signalValues.resize(static_cast<std::size_t>(signalSize));

        for (int i = 0; i < maskSize; ++i)
            in >> maskValues[static_cast<std::size_t>(i)];

        for (int i = 0; i < signalSize; ++i)
            in >> signalValues[static_cast<std::size_t>(i)];

        return FileStatus::Success;
    }

    inline FileStatus writeData1D(const std::string &fileName, const std::vector<float> &data)
    {
        std::ofstream out(fileName);
        if (!out)
            return FileStatus::NoFile;

        out << std::fixed << std::setprecision(2);
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            if (i)
                out << ' ';
            out << data[i];
        }
        out << '\n';
        return FileStatus::Success;
    }

    // --- 2D ---
    inline FileStatus getSizes2D(const std::string &fileName, int &maskSize, int &signalWidth, int &signalPitch, int &signalHeight) noexcept
    {
        std::ifstream in(fileName);
        if (!in)
            return FileStatus::NoFile;

        in >> maskSize;
        in >> signalWidth >> signalPitch >> signalHeight;
        return FileStatus::Success;
    }

    inline FileStatus getValues2D(const std::string &fileName, std::vector<float> &maskValues, std::vector<float> &signalValues)
    {
        std::ifstream in(fileName);
        if (!in)
            return FileStatus::NoFile;

        int maskSize = 0, signalWidth = 0, signalPitch = 0, signalHeight = 0;
        in >> maskSize;
        in >> signalWidth >> signalPitch >> signalHeight;

        maskValues.resize(static_cast<std::size_t>(maskSize * maskSize));
        signalValues.resize(static_cast<std::size_t>(signalHeight * signalPitch));

        for (std::size_t i = 0; i < maskValues.size(); ++i)
            in >> maskValues[i];

        for (std::size_t i = 0; i < signalValues.size(); ++i)
            in >> signalValues[i];

        return FileStatus::Success;
    }

    inline FileStatus writeData2D(const std::string &fileName, const std::vector<float> &data, int dataHeight, int dataPitch)
    {
        std::ofstream out(fileName);
        if (!out)
            return FileStatus::NoFile;

        out << std::fixed << std::setprecision(2);
        for (int i = 0; i < dataHeight; ++i)
        {
            for (int j = 0; j < dataPitch; ++j)
            {
                if (j)
                    out << ' ';
                out << data[static_cast<std::size_t>(i * dataPitch + j)];
            }
            out << '\n';
        }

        return FileStatus::Success;
    }

} // namespace conv
