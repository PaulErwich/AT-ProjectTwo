#pragma once

#include <iostream>
#include <array>
#include <filesystem>
#include <random>

#include "Room.h"

enum DOOR_TYPES
{
	DOOR4,
	DOOR3_LRT,
	DOOR3_LRB,
	DOOR3_LTB,
	DOOR3_RTB,
	DOOR2_LR,
	DOOR2_TB,
	DOOR2_LT,
	DOOR2_LB,
	DOOR2_RT,
	DOOR2_RB,
	DOOR1_L,
	DOOR1_R,
	DOOR1_T,
	DOOR1_B,
	DOOR_NONE,
	DOOR_OUT_OF_BOUNDS
};

static const std::string DOOR_TYPE_STRINGS[16] = {
	"door4", "door3LRT", "door3LRB", "door3LTB", "door3RTB",
	"door2LR", "door2TB", "door2LT", "door2LB", "door2RT", "door2RB",
	"door1L", "door1R", "door1T", "door1B", ""
};

class Dungeon
{
public:
	Dungeon();
	~Dungeon() = default;

	void BuildFullDungeon(bool useGen = false);
	void GenerateDungeonLayout();

	void render(sf::RenderWindow& window);
private:
	static const int MAP_WIDTH = 5;
	static const int MAP_HEIGHT = 5;

	std::string path_start = "data/dungeonImages/";

	std::array<Room, MAP_WIDTH * MAP_HEIGHT> dungeon;

	DOOR_TYPES genLayout[MAP_WIDTH * MAP_HEIGHT];

	//std::string fullLayout[MAP_WIDTH * MAP_HEIGHT] = {
	//"door2RB", "door3LRB", "door3LRB", "door2LR", "door1L",
	//"door1T", "door2TB", "door3RTB", "door2LR", "door2LB",
	//"door1R", "door4", "door4", "door3LRB", "door2LT",
	//"door2RB", "door2LT", "door2RB", "door3LTB", "door1B",
	//"door2RT", "door2LR", "door2LR", "door3LRT", "door2LT" };

	DOOR_TYPES fullLayout[MAP_WIDTH * MAP_HEIGHT] = {
		DOOR2_RB, DOOR3_LRB, DOOR3_LRB, DOOR2_LR, DOOR1_L,
		DOOR1_T, DOOR2_TB, DOOR3_RTB, DOOR2_LR, DOOR2_LB,
		DOOR1_R, DOOR4, DOOR4, DOOR3_LRB, DOOR2_LT,
		DOOR2_RB, DOOR2_LT, DOOR2_RB, DOOR3_LTB, DOOR1_B,
		DOOR2_RT, DOOR2_LR, DOOR2_LR, DOOR3_LRT, DOOR2_LT
	};
};