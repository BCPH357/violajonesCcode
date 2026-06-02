#include <cstdio>

int vj_cascade_top(int win_x, int win_y, int win_w, int win_h);

int main()
{
    int fails = 0;

    int r1 = vj_cascade_top(6, 6, 50, 50);
    printf("pass   (6,6,50,50) -> %d  expected 1\n", r1);
    if (r1 != 1) fails++;

    int r2 = vj_cascade_top(0, 0, 50, 50);
    printf("reject (0,0,50,50) -> %d  expected 0\n", r2);
    if (r2 != 0) fails++;

    if (fails == 0) { printf("C SIM PASS\n"); return 0; }
    else            { printf("C SIM FAIL (%d)\n", fails); return 1; }
}