# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/cshwstem/esp/esp-idf6/esp-idf/components/bootloader/subproject"
  "/home/cshwstem/CSW/TinyAuton/CODE/CHN/AIoTNode-CPP-CORE-CHN-AI-FW/build/bootloader"
  "/home/cshwstem/CSW/TinyAuton/CODE/CHN/AIoTNode-CPP-CORE-CHN-AI-FW/build/bootloader-prefix"
  "/home/cshwstem/CSW/TinyAuton/CODE/CHN/AIoTNode-CPP-CORE-CHN-AI-FW/build/bootloader-prefix/tmp"
  "/home/cshwstem/CSW/TinyAuton/CODE/CHN/AIoTNode-CPP-CORE-CHN-AI-FW/build/bootloader-prefix/src/bootloader-stamp"
  "/home/cshwstem/CSW/TinyAuton/CODE/CHN/AIoTNode-CPP-CORE-CHN-AI-FW/build/bootloader-prefix/src"
  "/home/cshwstem/CSW/TinyAuton/CODE/CHN/AIoTNode-CPP-CORE-CHN-AI-FW/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/cshwstem/CSW/TinyAuton/CODE/CHN/AIoTNode-CPP-CORE-CHN-AI-FW/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/cshwstem/CSW/TinyAuton/CODE/CHN/AIoTNode-CPP-CORE-CHN-AI-FW/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
