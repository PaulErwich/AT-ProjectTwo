#pragma once

#include <SFML/Graphics.hpp>
#include <iostream>
#include <tuple>
#include <queue>
#include <unordered_map>
#include "GameObject.h"
#include "Room.h"

using Location = std::tuple<int, int>;

namespace std
{
    template<> struct hash<Location>
    {
        std::size_t operator()(const Location& id) const noexcept
        {
            return std::get<0>(id) ^ std::get<1>(id);
        }
    };
}

using PQElement = std::pair<int, Location>;

std::vector<sf::Vector2i> pathfinding(sf::Vector2f xy, Room& map, GameObject& entity);
bool canPathfind(Room& map, Location start, Location end);

void AStar(Room& map, Location& start, Location& end,
    std::unordered_map<Location, Location>& came_from,
    std::unordered_map<Location, int>& cost_so_far);

std::vector<Location> getNeighbours(Room& map, Location start);

struct PriorityQueue
{
    std::priority_queue<PQElement, std::vector<PQElement>,
        std::greater<PQElement>> elements;

    inline bool empty() const {
        return elements.empty();
    }

    inline void put(Location item, int priority)
    {
        elements.emplace(priority, item);
    }

    Location get()
    {
        Location best_item = elements.top().second;
        elements.pop();
        return best_item;
    }
};

int heuristic(Location& a, Location& b);

std::vector<Location> reconstruct_path(
    Location start, Location end,
    std::unordered_map<Location, Location>& came_from);
