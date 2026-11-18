#ifndef NETTLM_DSE_LOG_H
#define NETTLM_DSE_LOG_H

#include <fstream>
#include <iostream>

#define LOG(message) do { std::cerr << message << std::endl; } while (false)
#define LOG2(message) do { std::cerr << message << std::endl; } while (false)

#define LOG3(message)                                                        \
    do {                                                                     \
        std::ofstream nettlm_log_stream("mapping_sa.log", std::ios::app);    \
        nettlm_log_stream << message << std::endl;                           \
    } while (false)

#define LOG_T(message) LOG(message)

#define LOG_EXP(filename, value)                                             \
    do {                                                                     \
        std::ofstream nettlm_log_stream((filename), std::ios::app);          \
        nettlm_log_stream << value << std::endl;                             \
    } while (false)

#define LOG_BY_J(summary, mapping)                                           \
    do {                                                                     \
        std::ofstream nettlm_log_stream("mapping_by_iteration.log",          \
                                        std::ios::app);                       \
        nettlm_log_stream << summary << std::endl << mapping << std::endl;   \
    } while (false)

#define LOG_HOUR(summary, mapping)                                           \
    do {                                                                     \
        std::ofstream nettlm_log_stream("mapping_hour.log", std::ios::app);  \
        nettlm_log_stream << summary << std::endl << mapping << std::endl;   \
    } while (false)

#define LOG_MINUTE(summary, mapping)                                         \
    do {                                                                     \
        std::ofstream nettlm_log_stream("mapping_minute.log",                \
                                        std::ios::app);                       \
        nettlm_log_stream << summary << std::endl << mapping << std::endl;   \
    } while (false)

#endif
