cmake_minimum_required(VERSION 3.25)
project(vertirta_consumer_smoke VERSION 1.0.0 LANGUAGES CXX)

find_package(VertiRTA CONFIG REQUIRED COMPONENTS flight_core)

add_executable(consumer_smoke consumer_smoke.cpp)
target_link_libraries(consumer_smoke PRIVATE VertiRTA::flight_core)
target_compile_features(consumer_smoke PRIVATE cxx_std_17)
target_compile_options(consumer_smoke PRIVATE -Wall -Wextra -Wpedantic -Werror)

enable_testing()
add_test(NAME vertirta_consumer_smoke COMMAND consumer_smoke)
