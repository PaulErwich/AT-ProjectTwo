#pragma once

#include <iostream>
#include <SFML/Graphics.hpp>

#include "Tile.h"

class Room
{
public:
	Room();
	~Room() = default;

	bool init(std::string path, bool build_map = true, int room_pos = 0);

	void buildMap();

	void clearMap()
	{
		layer.clear();
	}

	void update(float dt);
	void render(sf::RenderWindow& window);

	int getTileCost(int x, int y) { return cost_map[y * WIDTH + x]; };

	static sf::Vector2i world(std::tuple<int, int> tile_coords);
	static std::tuple<int, int> tile(sf::Vector2f point);

	static int getWidth() { return WIDTH; }
	static int getHeight() { return HEIGHT; }

	const int* getCostMap() { return cost_map; }
private:
	const int SCALE = 1;
	static const int WIDTH = 32;
	static const int HEIGHT = 32;

	int room_position;

	sf::Texture tilemap;
	sf::Image image;

	std::vector<std::unique_ptr<Tile>> layer;

	int cost_map[WIDTH * HEIGHT];
};