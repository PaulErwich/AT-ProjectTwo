#pragma once

#include <iostream>
#include <SFML/Graphics.hpp>

#include "Tile.h"

enum DOOR_TYPES
{
	DOOR4,
	DOOR3_LEFT_RIGHT_TOP,
	DOOR3_LEFT_RIGHT_BOTTOM,
	DOOR3_LEFT_TOP_BOTTOM,
	DOOR3_RIGHT_TOP_BOTTOM,
	DOOR2_LEFT_RIGHT,
	DOOR2_TOP_BOTTOM,
	DOOR2_LEFT_TOP,
	DOOR2_LEFT_BOTTOM,
	DOOR2_RIGHT_TOP,
	DOOR2_RIGHT_BOTTOM,
	DOOR1_LEFT,
	DOOR1_RIGHT,
	DOOR1_TOP,
	DOOR1_BOTTOM
};

class Room
{
public:
	Room();
	~Room() = default;

	bool init(std::string path, bool build_map = true);

	void buildMap();

	void update(float dt);
	void render(sf::RenderWindow& window);

	int getTileCost(int x, int y) { return cost_map[y * WIDTH + x]; };

	static sf::Vector2i world(std::tuple<int, int> tile_coords);
	static std::tuple<int, int> tile(sf::Vector2f point);

	static int getWidth() { return WIDTH; }
	static int getHeight() { return HEIGHT; }

	const int* getCostMap() { return cost_map; }
private:
	const int SCALE = 2;
	static const int WIDTH = 32;
	static const int HEIGHT = 32;

	sf::Texture tilemap;
	sf::Image image;

	std::vector<std::unique_ptr<Tile>> layer;

	int cost_map[WIDTH * HEIGHT];
};