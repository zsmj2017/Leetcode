int maxAreaOfIsland(vector<vector<int>> &grid) {
    if (grid.empty() || grid[0].empty()) {
        return 0;
    }
    int res = 0;
    for (int i = 0; i < grid.size(); ++i) {
        for (int j = 0; j < grid[0].size(); ++j) {
            res = max(res, aux(i, j, grid));
        }
    }
    return res;
}
int aux(int x, int y, vector<vector<int>> &grid) {
    if (x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size() && grid[x][y] == 1) {
        grid[x][y] = 0;
        return 1 + aux(x - 1, y, grid) + aux(x + 1, y, grid) + aux(x, y - 1, grid) + aux(x, y + 1, grid);
    }
    return 0;
}