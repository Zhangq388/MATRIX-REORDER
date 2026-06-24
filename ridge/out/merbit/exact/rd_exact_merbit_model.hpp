#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace rd_exact_merbit_model
{
    inline constexpr std::size_t kFeatureCount = 10;

    // Input order must be r1, r2, ..., r10 from the exact RD extractor.
    inline constexpr std::array<const char*, kFeatureCount> kFeatureNames = {
        "r1",
        "r2",
        "r3",
        "r4",
        "r5",
        "r6",
        "r7",
        "r8",
        "r9",
        "r10"
    };

    inline constexpr std::array<double, kFeatureCount> kImputerMedian = {
        9.66271499999999950e-02,
        1.15343500000000002e-01,
        2.73571999999999982e-02,
        2.37909499999999996e-01,
        8.74864999999999948e-02,
        6.53402500000000025e-02,
        5.64822999999999974e-03,
        0.00000000000000000e+00,
        0.00000000000000000e+00,
        0.00000000000000000e+00
    };

    inline constexpr std::array<double, kFeatureCount> kScalerMean = {
        1.76524068954938274e-01,
        1.66730340391358006e-01,
        5.15886969800493783e-02,
        2.51083424741286920e-01,
        1.08608703998013700e-01,
        1.07168189661850613e-01,
        7.06814809161975394e-02,
        3.34103287153246983e-02,
        2.01213325583481442e-02,
        1.40834445953086414e-02
    };

    inline constexpr std::array<double, kFeatureCount> kScalerScale = {
        1.94184408296748490e-01,
        1.67530229256701080e-01,
        7.24449943879360569e-02,
        1.77952694288075031e-01,
        1.04068037168718860e-01,
        1.16539888858830210e-01,
        1.08348084136699149e-01,
        8.18006438381082973e-02,
        6.78629705325223265e-02,
        6.45444829943046666e-02
    };

    inline constexpr std::array<double, kFeatureCount> kScaledCoef = {
        -6.54719271634739697e-02,
        -4.22408258353547394e-02,
        -1.78439564123868653e-02,
        -9.65255307966482977e-03,
        2.78430239162094049e-02,
        -1.30943276679122700e-03,
        -5.96365663198338697e-03,
        -2.33941362544576037e-02,
        1.92918315414425584e-01,
        1.47574899030913032e-01
    };

    inline constexpr double kScaledIntercept =
        2.42110819725922966e-01;

    // StandardScaler has been folded into these parameters.
    inline constexpr std::array<double, kFeatureCount> kRawCoef = {
        -3.37163666937775774e-01,
        -2.52138530596949795e-01,
        -2.46310411963512277e-01,
        -5.42422418400644937e-02,
        2.67546354036343470e-01,
        -1.12359191313233468e-02,
        -5.50416436017386462e-02,
        -2.85989634760784428e-01,
        2.84276261266771879e+00,
        2.28640609057066690e+00
    };

    inline constexpr double kRawIntercept =
        2.66184683719140058e-01;

    inline double PredictLogSpeedup(
        const std::array<double, kFeatureCount>& rd_feature)
    {
        double value = kRawIntercept;
        for(std::size_t i = 0; i < kFeatureCount; ++i)
        {
            value += kRawCoef[i] * rd_feature[i];
        }
        return value;
    }

    inline double PredictSpeedup(
        const std::array<double, kFeatureCount>& rd_feature)
    {
        return std::exp(PredictLogSpeedup(rd_feature));
    }

    inline double PredictSpeedupWithPreprocess(
        std::array<double, kFeatureCount> rd_feature,
        const std::array<bool, kFeatureCount>& missing)
    {
        double value = kScaledIntercept;
        for(std::size_t i = 0; i < kFeatureCount; ++i)
        {
            const double x = missing[i] ? kImputerMedian[i] : rd_feature[i];
            const double standardized = (x - kScalerMean[i]) / kScalerScale[i];
            value += kScaledCoef[i] * standardized;
        }
        return std::exp(value);
    }
}
