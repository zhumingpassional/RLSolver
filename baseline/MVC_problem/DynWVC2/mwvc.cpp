#include "mwvc.h"
#include <sstream>

int main(int argc, char *argv[])
{
    uint seed;
    
    if (argc == 1)
    {
        cout << "Minimum Weighted Vertex Cover Problem solver." << endl;
        cout << "Usage: ./mwvc [Graph file] [Seed] [Cutoff time] [CC mode]" << endl;
        return 1;
    }

    if (argc < 5)
    {
        cerr << "Missing argument(s)." << endl;
        cout << "Usage: ./mwvc [Graph file] [Seed] [Cutoff time] [CC mode]" << endl;
        return 1;
    }

    stringstream ss;
    ss << argv[2];
    ss >> seed;
    ss.clear();
    ss << argv[3];
    ss >> cutoff_time;
    ss.clear();
    ss << argv[4];
    ss >> mode;
    ss.clear();

    if (BuildInstance(argv[1]) != 0)
    {
        cerr << "Open instance file failed." << endl;
        return 1;
    }

    if (seed < 0U || seed > ~0U)
    {
        seed = 10;
    }

    if (cutoff_time < 0 || cutoff_time > (int)(~0U>>1))
    {
        cutoff_time = 1000;
    }

    if (mode < 0 || mode > 3)
    {
        mode = 0;
    }

    srand(seed);

    cout << argv[1];

    start = chrono::steady_clock::now();

    ConstructVC();
    LocalSearch();

    if (CheckSolution() == 1)
    {
        cout << "best_w:" << best_weight << " best_time: " << best_comp_time <<endl;
    }
    else
    {
        cout << ", the solution is wrong." << endl;
    }

    FreeMemory();

    return 0;
}
