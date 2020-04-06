cmake_minimum_required(VERSION 3.3)

project(qp C)

option(CXX "enable C++ compilation" ON)
if(CXX)
    enable_language(CXX)
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(GUROBI REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)
find_package(glpk REQUIRED)

set(CMAKE_SHARED_MODULE_PREFIX "")

include_directories(${GUROBI_INCLUDE_DIRS})
include_directories(${glpk_INCLUDE_DIRS})

set(GUROBI_LIBRARIES "${GUROBI_CXX_LIBRARY};${GUROBI_LIBRARY}")

# create the lib
#add_executable(qpg qpg.cpp)
add_library(qpg MODULE qpg.cpp)


#~ target_link_libraries(solveLP ${PYTHON_LIBRARIES})
target_link_libraries(qpg ${GUROBI_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${glpk_LIBRARIES})
target_include_directories(qpg PRIVATE ${PYTHON_INCLUDE_DIRS})


