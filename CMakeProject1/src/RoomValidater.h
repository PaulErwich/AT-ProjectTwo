#pragma once

#include <iostream>
#include <SFML/Graphics.hpp>

#include "Room.h"
#include "Pathfinding.h"

class RoomValidator
{
public:
	RoomValidator();
	~RoomValidator() = default;

	bool validateRoom(std::string imagePath);

private:
	void moveFile(std::string inPath, std::string outPath);

	const std::string PATH = "data/genOutput/samples/";
	Room room;

	const static int WIDTH = 32;
	const static int HEIGHT = 32;
};