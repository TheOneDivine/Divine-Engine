#include "App.hpp"

int main() {
   try {
      // initialize the app (implicit)
      App app;

      // main loop
      app.run();

   }// app cleanup is handled here (implicit) by the destructor

   // catch any exceptions and process them
   catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      // if any exceptions are caught, app is considered failure
      return EXIT_FAILURE;
   }
   // else app is considered success
   return EXIT_SUCCESS;
}