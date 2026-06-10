#include "app.h"
#include <filesystem>

int main(int argc, char **argv) {
  const std::filesystem::path appDir =
      std::filesystem::absolute(argv[0]).parent_path();

  App app(appDir);
  app.run();
  return EXIT_SUCCESS;
}
