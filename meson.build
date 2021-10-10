project('TTempSmooth', 'cpp',
  default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++17'],
  license: 'GPL-3.0-or-later',
  meson_version: '>=0.51.0',
  version: '3'
)

cxx = meson.get_compiler('cpp')

gcc_syntax = cxx.get_argument_syntax() == 'gcc'

if get_option('buildtype') == 'release'
  add_project_arguments(gcc_syntax ? ['-fno-math-errno', '-fno-trapping-math'] : '/GS-', language: 'cpp')
endif

if gcc_syntax
  vapoursynth_dep = dependency('vapoursynth', version: '>=55').partial_dependency(compile_args: true, includes: true)
  install_dir = vapoursynth_dep.get_variable(pkgconfig: 'libdir') / 'vapoursynth'
else
  vapoursynth_dep = dependency('', required: false)
  install_dir = get_option('libdir') / 'vapoursynth'
endif

sources = [
  'TTempSmooth/TTempSmooth.cpp'
]

if host_machine.cpu_family().startswith('x86') and gcc_syntax
  add_project_arguments('-mfpmath=sse', '-msse2', language: 'cpp')
endif

shared_module('ttempsmooth', sources,
  dependencies: vapoursynth_dep,
  install: true,
  install_dir: install_dir,
  gnu_symbol_visibility: 'hidden'
)
