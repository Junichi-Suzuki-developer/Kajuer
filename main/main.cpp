#include <mkl.h>
#include "Kajuer.h"

int main(int argc, char *argv[])
{
    //To gain performance, we call vmlSetMode here.
    vmlSetMode( VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE );

    Kajuer::Runner::Runner runner;

    runner.run(argc, argv);

    return 0;
}
