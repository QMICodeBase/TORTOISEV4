# Wrap a .wgsl file in a C++ raw string literal so shaders are compiled into the
# binary - no runtime asset paths to get wrong.
#   cmake -DIN=x.wgsl -DOUT=x.wgsl.h -DVAR=kX -P embed_wgsl.cmake
file(READ ${IN} _src)
# Neither WGSL nor the .metal sources use #include for this; the metric shaders
# share a prelude, prepended here. The prelude has the same extension as its
# consumers, so one script serves both backends.
get_filename_component(_dir ${IN} DIRECTORY)
get_filename_component(_base ${IN} NAME)
get_filename_component(_ext ${IN} EXT)
if(_base MATCHES "^metric_" AND NOT _base STREQUAL "metric_common${_ext}")
    file(READ ${_dir}/metric_common${_ext} _prelude)
    set(_src "${_prelude}\n${_src}")
endif()
if(_src MATCHES "\\)WGSL")
    message(FATAL_ERROR "${IN} contains the raw-string terminator )WGSL")
endif()
get_filename_component(_name ${IN} NAME)
file(WRITE ${OUT}
     "// Generated from ${_name} - do not edit.\n"
     "#pragma once\n"
     "static const char *${VAR} = R\"WGSL(\n${_src})WGSL\";\n")
