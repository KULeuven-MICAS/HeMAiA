#include <array>
#include <cmath>
#include <cstdint>
#include <cassert>

namespace QuantizedMath {

    // Corresponds to DT_QUANT_RESULT in the original code
    using QuantResult = uint8_t;
    // Corresponds to DT_QKT_ROW in the original code (an array of 64 elements)
    using SoftmaxRow = std::array<QuantResult, 64>;

    /**
     * @brief Polynomial fitting function used to approximate the exponential.
     * Uses a quadratic polynomial: y = a*(p + b)^2 + c
     */
    inline int i_poly(int p) {
        float p_fp = static_cast<float>(p);
        const float a = 0.3585f;
        const float b = 1.353f;
        const float c = 0.344f;
        
        return static_cast<int>(a * (p_fp + b) * (p_fp + b) + c);
    }

    /**
     * @brief Computes the quantized exponential approximation.
     */
    inline int i_exp(int q, int S) {
        // Calculate ln(2) using the standard library and scale by S
        const int ln2 = static_cast<int>(std::log(2.0f) * S); 
        int q_ln2 = ln2 / S; 
        
        // Division-by-zero protection: return 2^q directly if q_ln2 is 0
        if (q_ln2 == 0) {
            return 1 << q; 
        }

        int z = -q / q_ln2; 
        int p = q + z * q_ln2; 

        // Approximate exp(p) using the polynomial and bit-shift by z
        int q_L = i_poly(p / S);
        return q_L >> z; 
    }

    /**
     * @brief Softmax based on purely integer approximation.
     * 
     * @param vec Input vector (size 64).
     * @param max_value The maximum value in the vector, subtracted for numerical stability.
     * @return A uint8_t array containing the normalized exponential results mapped to the 0-255 range.
     */
    inline SoftmaxRow softmax(const std::array<int, 64>& vec, int max_value) {
        SoftmaxRow result = {0};
        int S = 1; // Scaling factor
        int sum_q_exp = 0;
        std::array<int, 64> q_exp_list = {0};
        
        // 1. Subtract the max value (for numerical stability) and compute the approximate exponential
        for (size_t i = 0; i < vec.size(); ++i) {
            int q_tilde = vec[i] - max_value; 
            q_exp_list[i] = i_exp(q_tilde, S);
            sum_q_exp += q_exp_list[i]; 
        }

        // Defensive programming: ensure the sum is not zero to prevent division-by-zero crashes
        assert((sum_q_exp != 0) && "Division by zero: sum_q_exp is 0");

        // 2. Normalize and map the results to the 8-bit (0~255) range
        for (size_t i = 0; i < vec.size(); ++i) {
            result[i] = static_cast<QuantResult>(q_exp_list[i] * 255 / sum_q_exp); 
        }

        return result;
    }

} // namespace QuantizedMath