#pragma once

#include <SFML/Graphics.hpp>
#include <iostream>
#include <tuple>
#include "GameObject.h"

class Tile : public GameObject
{
public:
	Tile(sf::Texture& _texture, int x, int y, int texture_num, int _cost, int scale);
	~Tile() = default;

	// get X, Y coordinates of the tile
	std::tuple<int, int> getCoords();

	// get cost of tile
	int getCost() const;

protected:
	// Represents X, Y of the tile
	// This is for pathfinding
	std::tuple<int, int> tile_coords;
	int cost;
};
