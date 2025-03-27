#include "Pathfinding.h"

std::vector<sf::Vector2i> pathfinding(sf::Vector2f xy, Room& map, GameObject& entity)
{
    Location end(map.tile(xy));

    int tile_cost = map.getTileCost(std::get<0>(end), std::get<1>(end));

    std::vector<sf::Vector2i> tiles_to_visit;
    tiles_to_visit.reserve(100);

    sf::Vector2f player_pos(
        entity.getMin().x + entity.getWidth() / 2,
        entity.getMin().y + entity.getHeight() / 2);
    Location start(map.tile(player_pos));

    std::unordered_map<Location, Location> came_from;
    std::unordered_map<Location, int> cost_so_far;

    if (tile_cost < 1)
    {
        //std::cout << "finding" << std::endl;
        AStar(map, start, end, came_from, cost_so_far);

        for (auto loc : reconstruct_path(start, end, came_from))
        {
            //std::cout << std::get<0>(loc) << " " << std::get<1>(loc) << "\n";
            tiles_to_visit.push_back(map.world(loc));
        }
    }

    return tiles_to_visit;
}

bool canPathfind(Room& map, Location start, Location end)
{
    int tile_cost = map.getTileCost(std::get<0>(end), std::get<1>(end));

    std::vector<sf::Vector2i> tiles_to_visit;
    tiles_to_visit.reserve(100);

    std::unordered_map<Location, Location> came_from;
    std::unordered_map<Location, int> cost_so_far;

    if (tile_cost < 1)
    {
        //std::cout << "finding" << std::endl;
        AStar(map, start, end, came_from, cost_so_far);

        for (auto loc : reconstruct_path(start, end, came_from))
        {
            //std::cout << std::get<0>(loc) << " " << std::get<1>(loc) << "\n";
            tiles_to_visit.push_back(map.world(loc));
        }
    }

    if (tiles_to_visit.size() != 0)
        return true;
    return false;
}

void AStar(Room& map, Location& start, Location& end,
    std::unordered_map<Location, Location>& came_from,
    std::unordered_map<Location, int>& cost_so_far)
{
    int width = map.getWidth();
    int height = map.getHeight();

    PriorityQueue frontier;

    frontier.put(start, 0);

    came_from[start] = start;
    cost_so_far[start] = 0;

    while (!frontier.empty())
    {
        Location current = frontier.get();

        if (current == end)
        {
            break;
        }

        for (auto node : getNeighbours(map, current))
        {
            int new_cost = cost_so_far[current] +
                map.getTileCost(std::get<0>(node), std::get<1>(node));

            if (cost_so_far.find(node) == cost_so_far.end() ||
                new_cost < cost_so_far[node])
            {
                cost_so_far[node] = new_cost;
                int priority = new_cost + heuristic(node, end);
                frontier.put(node, priority);
                came_from[node] = current;
            }
        }
    }
}

std::vector<Location> getNeighbours(Room& map, Location start)
{
    std::vector<Location> neighbours;
    neighbours.reserve(4);

    std::tuple<int, int> DIRS[4] = {
      {1, 0},
      {-1, 0},
      {0, 1},
      {0, -1}
    };

    for (auto direction : DIRS)
    {
        Location current = std::make_tuple
        (std::get<0>(start) + std::get<0>(direction),
            std::get<1>(start) + std::get<1>(direction));

        if (std::get<0>(current) > map.getWidth() - 1 || std::get<0>(current) < 0 ||
            std::get<1>(current) > map.getHeight() - 1 || std::get<1>(current) < 0) continue;

        if (map.getTileCost(std::get<0>(current), std::get<1>(current)) > 0) continue;

        neighbours.push_back(current);
    }

    return neighbours;
}

int heuristic(Location& a, Location& b)
{
    return (abs(std::get<0>(a) - abs(std::get<0>(b))) +
        abs(std::get<1>(a) - abs(std::get<1>(b))));
}

std::vector<Location> reconstruct_path(
    Location start, Location end,
    std::unordered_map<Location, Location>& came_from)
{
    std::vector<Location> path;
    path.reserve(100);

    Location current = end;

    if (came_from.find(end) == came_from.end())
    {
        return path; // nothing found
    }

    while (current != start)
    {
        path.push_back(current);
        current = came_from[current];
    }
    //path.push_back(start); // adds start
    std::reverse(path.begin(), path.end());

    return path;
}