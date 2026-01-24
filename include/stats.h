#ifndef STATS_H
#define STATS_H

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

inline double sum(const std::vector<double>& a)
{
    return std::accumulate(a.begin(), a.end(), 0.0);
}

inline double mean(const std::vector<double>& a)
{
    if (a.empty()) return 0.0;
    return sum(a) / a.size();
}

inline double sqsum(const std::vector<double>& a)
{
    double s = 0;
    for (double val : a)
    {
        s += val * val;
    }
    return s;
}

inline double stdev(const std::vector<double>& nums)
{
    if (nums.empty()) return 0.0;
    double N = (double)nums.size();
    double s = sum(nums);
    double ss = sqsum(nums);
    double var = (ss / N) - pow(s / N, 2);
    return sqrt(std::max(0.0, var));
}

inline double pearsoncoeff(const std::vector<double>& X, const std::vector<double>& Y)
{
    if (X.size() != Y.size() || X.empty()) return 0.0;

    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    size_t n = X.size();

    for (size_t i = 0; i < n; ++i)
    {
        sumX += X[i];
        sumY += Y[i];
        sumXY += X[i] * Y[i];
        sumX2 += X[i] * X[i];
        sumY2 += Y[i] * Y[i];
    }

    double denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    if (denominator == 0) return 0.0;

    return (n * sumXY - sumX * sumY) / denominator;
}

#endif
