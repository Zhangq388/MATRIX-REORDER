#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace rd_sampled_csr_model
{
    inline constexpr std::size_t kFeatureCount = 10;

    // Input order must be r1, r2, ..., r10 from the sampled RD extractor.
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
        -6.28390700964915755e-02,
        -5.03484292953999477e-02,
        4.31497477198774344e-04,
        -1.06990378674301778e-02,
        4.64307771452045770e-02,
        -1.50315946557232124e-02,
        3.25151857123279764e-02,
        -3.72982548746619777e-02,
        1.20084702312841615e-01,
        1.63218327522822865e-01
    };

    inline constexpr double kScaledIntercept =
        2.47726915440044798e-01;

    // StandardScaler has been folded into these parameters.
    inline constexpr std::array<double, kFeatureCount> kRawCoef = {
        -3.23804724610474881e-01,
        -3.00486639537123124e-01,
        5.97989718749250988e-03,
        -6.04430805086655781e-02,
        4.52291486900478046e-01,
        -1.28875868013209188e-01,
        2.97368471773662213e-01,
        -4.54258534668076253e-01,
        1.76762766759557954e+00,
        2.45978651843712059e+00
    };

    inline constexpr double kRawIntercept =
        2.56523531597976584e-01;

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
