#pragma once
#include "inc.cuh"
namespace rd_merbit_model
{
    inline constexpr std::size_t kFeatureCount = 10;

    // Input order must be r1, r2, ..., r10 from the sampled RD extractor.
    inline constexpr std::array<const char*, kFeatureCount> kFeatureNames = {"r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"};

    inline constexpr std::array<double, kFeatureCount> kImputerMedian = {
        9.77429499999999951e-02,
        1.15494500000000000e-01,
        2.74992000000000014e-02,
        2.36329000000000011e-01,
        8.61308000000000074e-02,
        6.44094500000000070e-02,
        5.81218500000000009e-03,
        0.00000000000000000e+00,
        0.00000000000000000e+00,
        0.00000000000000000e+00
    };

    inline constexpr std::array<double, kFeatureCount> kScalerMean = {
        1.76441841357901252e-01,
        1.66786440354320997e-01,
        5.14436632791358053e-02,
        2.49631313800000004e-01,
        1.07480775665555556e-01,
        1.07620488881728413e-01,
        7.15868577962963104e-02,
        3.38007868471604939e-02,
        2.02931203872839527e-02,
        1.49147190641975309e-02
    };

    inline constexpr std::array<double, kFeatureCount> kScalerScale = {
        1.94064710365436016e-01,
        1.67556299251633567e-01,
        7.21580093552929153e-02,
        1.77010135442985606e-01,
        1.02656756737544291e-01,
        1.16636224356467907e-01,
        1.09343083745194233e-01,
        8.21079892355034563e-02,
        6.79355186130273514e-02,
        6.63546719601208340e-02
    };

    inline constexpr std::array<double, kFeatureCount> kScaledCoef = {
        -6.52556894462520132e-02,
        -4.28585922763317709e-02,
        -1.74969311762466044e-02,
        -1.06739269643762909e-02,
        2.87990055385227359e-02,
        -1.91283223810189860e-03,
        -5.06737395751141566e-03,
        -2.47685247239517459e-02,
        1.87099256711209222e-01,
        1.52900272760692069e-01
    };

    inline constexpr double kScaledIntercept =
        2.42110819725922966e-01;

    // StandardScaler has been folded into these parameters.
    inline constexpr std::array<double, kFeatureCount> kRawCoef = {
        -3.36257371695098284e-01,
        -2.55786219126070413e-01,
        -2.42480790872360380e-01,
        -6.03012191232085021e-02,
        2.80536873107643969e-01,
        -1.63999842129305441e-02,
        -4.63437995705341804e-02,
        -3.01657913615571149e-01,
        2.75407122122610781e+00,
        2.30428797617423475e+00
    };

    inline constexpr double kRawIntercept = 2.66499564016047963e-01;

    inline double PredictLogSpeedup(const std::array<double, kFeatureCount>& rd_feature)
    {
        double value = kRawIntercept;
        for(std::size_t i = 0; i < kFeatureCount; ++i)
        {
            value += kRawCoef[i] * rd_feature[i];
        }
        return value;
    }

    inline double PredictSpeedup(const std::array<double, kFeatureCount>& rd_feature)
    {
        return std::exp(PredictLogSpeedup(rd_feature));
    }

    inline double PredictSpeedupWithPreprocess(std::array<double, kFeatureCount> rd_feature, const std::array<bool, kFeatureCount>& missing)
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