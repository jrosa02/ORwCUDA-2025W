#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace histogram_utils
{

	// Read entire file into a vector<uint8_t>. Throws on failure.
	inline std::vector<unsigned char> readFile(const std::filesystem::path &path)
	{
		std::ifstream ifs(path);
		if (!ifs)
			throw std::runtime_error("failed to open file for reading: " + path.string());

		std::vector<unsigned char> buffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
		if (buffer.empty())
			throw std::runtime_error("file is empty or could not be read: " + path.string());

		return buffer;
	}

	// Write a histogram to a text file with human-readable bins.
	// histogram: vector of counts (one per bin). nLetters defaults to 26 (a..z).
	inline void writeHistogram(const std::filesystem::path &path, const std::vector<unsigned int> &histogram, int nBins, int nLetters = 26)
	{
		if (nBins <= 0)
			throw std::invalid_argument("nBins must be positive");

		std::ofstream ofs(path);
		if (!ofs)
			throw std::runtime_error("failed to open file for writing: " + path.string());

		// ceil(nLetters / nBins)
		int binWidth = static_cast<int>(std::ceil(static_cast<float>(nLetters) / nBins));
		binWidth = std::max(1, binWidth);

		if (binWidth == 1)
		{
			for (int i = 0; i < nBins; ++i)
			{
				char label = static_cast<char>('a' + i);
				ofs << label << ": " << (i < static_cast<int>(histogram.size()) ? histogram[i] : 0u) << '\n';
			}
		}
		else
		{
			for (int i = 0; i * binWidth < nLetters; ++i)
			{
				char start = static_cast<char>('a' + i * binWidth);
				int endIdx = (i + 1) * binWidth <= nLetters ? (i + 1) * binWidth - 1 : nLetters - 1;
				char end = static_cast<char>('a' + endIdx);
				ofs << start << '-' << end << ": " << (i < static_cast<int>(histogram.size()) ? histogram[i] : 0u) << '\n';
			}
		}
	}

} // namespace histogram_utils
