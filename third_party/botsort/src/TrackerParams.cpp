// Motion-only build: configuration comes from the calculator, never from .ini
// files, so load_config is a never-executed stub kept only so its symbol stays
// defined (BoTSORT.cpp takes its address). Drops the INIReader dependency.
#include "TrackerParams.h"

TrackerParams TrackerParams::load_config(const std::string& /*config_path*/) {
  return TrackerParams{};
}
