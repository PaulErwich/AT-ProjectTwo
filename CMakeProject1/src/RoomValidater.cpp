#include "RoomValidater.h"
#include <filesystem>


RoomValidator::RoomValidator()
{

}

bool RoomValidator::validateRoom(std::string imagePath)
{
	if (!room.init(imagePath, false))
		return false;

	// Check what doors the generated room has
	bool left = !room.getCostMap()[14 * WIDTH + 0] && !room.getCostMap()[15 * WIDTH + 0] &&
		!room.getCostMap()[16 * WIDTH + 0] && !room.getCostMap()[17 * WIDTH + 0];

	bool right = !room.getCostMap()[14 * WIDTH + 31] && !room.getCostMap()[15 * WIDTH + 31] &&
		!room.getCostMap()[16 * WIDTH + 31] && !room.getCostMap()[17 * WIDTH + 31];

	bool top = !room.getCostMap()[0 * WIDTH + 14] && !room.getCostMap()[0 * WIDTH + 15] &&
		!room.getCostMap()[0 * WIDTH + 16] && !room.getCostMap()[0 * WIDTH + 17];

	bool bottom = !room.getCostMap()[31 * WIDTH + 14] && !room.getCostMap()[31 * WIDTH + 15] &&
		!room.getCostMap()[31 * WIDTH + 16] && !room.getCostMap()[31 * WIDTH + 17];

	std::cout << top << " " << bottom << " " << left << " " << right << " " << std::endl;

	// Now check if they're accessible
	if (left && right && top && bottom)
	{
		if (canPathfind(room, { 0, 14 }, { 31, 14 }) && canPathfind(room, { 14, 0 }, { 14, 31 }))
		{
			moveFile(imagePath, "data/dungeonImages/door4LRTB/");
			return true;
		}
	}
	if (left && right && top)
	{
		if (canPathfind(room, { 0, 14 }, { 31, 14 }) && canPathfind(room, { 0, 14 }, { 14, 0 }))
		{
			// Room valid - left_right_top
			moveFile(imagePath, "data/dungeonImages/door3LRT/");
			return true;
		}
	}
	if (left && right && bottom)
	{
		if (canPathfind(room, { 0, 14 }, { 31, 14 }) && canPathfind(room, { 0, 14 }, { 14, 31 }))
		{
			// Room valid - left_right_bottom
			moveFile(imagePath, "data/dungeonImages/door3LRB/");
			return true;
		}
	}
	if (left && top && bottom)
	{
		if (canPathfind(room, { 0, 14 }, { 14, 0 }) && canPathfind(room, { 0, 14 }, { 14, 31 }))
		{
			// left_top_bottom
			moveFile(imagePath, "data/dungeonImages/door3LTB/");
			return true;
		}
	}
	if (right && top && bottom)
	{
		if (canPathfind(room, { 31, 14 }, { 14, 0 }) && canPathfind(room, { 31, 14 }, { 14, 31 }))
		{
			// right_top_bottom
			moveFile(imagePath, "data/dungeonImages/door3RTB/");
			return true;
		}
	}
	if (left && right)
	{
		if (canPathfind(room, { 0, 14 }, { 31, 14 }))
		{
			// left_right
			moveFile(imagePath, "data/dungeonImages/door2LR/");
			return true;
		}
	}
	if (top && bottom)
	{
		if (canPathfind(room, { 14, 0 }, { 14, 31 }))
		{
			// top_bottom
			moveFile(imagePath, "data/dungeonImages/door2TB/");
			return true;
		}
	}
	if (left && top)
	{
		if (canPathfind(room, { 0, 14 }, { 14, 0 }))
		{
			// left_top
			moveFile(imagePath, "data/dungeonImages/door2LT/");
			return true;
		}
	}
	if (left && bottom)
	{
		if (canPathfind(room, { 0, 14 }, { 14, 31 }))
		{
			// left_bottom
			moveFile(imagePath, "data/dungeonImages/door2LB/");
			return true;
		}
	}
	if (right && top)
	{
		if (canPathfind(room, { 31, 14 }, { 14, 0 }))
		{
			// right_top
			moveFile(imagePath, "data/dungeonImages/door2RT/");
			return true;
		}
	}
	if (right && bottom)
	{
		if (canPathfind(room, { 31, 14 }, { 14, 31 }))
		{
			// right_bottom
			moveFile(imagePath, "data/dungeonImages/door2RB/");
			return true;
		}
	}

	if (left && !right && !top && !bottom)
	{
		// left door
		moveFile(imagePath, "data/dungeonImages/door1L/");
		return true;
	}
	if (right && !left && !top && !bottom)
	{
		// right door
		moveFile(imagePath, "data/dungeonImages/door1R/");
		return true;
	}

	if (top && !left && !right && !bottom)
	{
		// top door
		moveFile(imagePath, "data/dungeonImages/door1T/");
		return true;
	}
	if (bottom && !left && !right && !top)
	{
		// bottom door
		moveFile(imagePath, "data/dungeonImages/door1B/");
		return true;
	}

	moveFile(imagePath, "data/dungeonImages/invalid/");

	return true;
}

void RoomValidator::moveFile(std::string inPath, std::string outPath)
{
	auto dirIter = std::filesystem::directory_iterator(outPath);
	int fileCount = 0;

	for (auto& entry : dirIter)
	{
		//std::cout << entry.path().filename();
		if (entry.is_regular_file())
		{
			++fileCount;
		}
	}
	std::filesystem::rename(inPath, outPath + std::to_string(fileCount) + ".png");
	std::cout << "New room in: " + outPath + "\n";
}