#include "Dungeon.h"

#include <algorithm>

#define random(min, max) rand() % (max + 1 - min) + min

Dungeon::Dungeon()
{

}

void Dungeon::BuildFullDungeon(bool useGen)
{	
	
	DOOR_TYPES layout[MAP_WIDTH * MAP_HEIGHT];
	for (int i = 0; i < MAP_WIDTH * MAP_HEIGHT; i++)
	{
		layout[i] = fullLayout[i];
	}

	if (useGen)
	{
		for (int i = 0; i < MAP_WIDTH * MAP_HEIGHT; i++)
		{
			layout[i] = genLayout[i];
		}
	}

	for (int i = 0; i < 25; i++)
	{
		if (layout[i] == DOOR_NONE)
		{
			dungeon[i].clearMap();
			continue;
		}

		int fileCount = 0;
		auto dirIter = std::filesystem::directory_iterator(path_start + DOOR_TYPE_STRINGS[layout[i]] + "/");
		for (auto& entry : dirIter)
		{
			if (entry.is_regular_file())
				fileCount++;
		}

		dungeon[i].init(path_start + DOOR_TYPE_STRINGS[layout[i]] + "/" + std::to_string(random(0, fileCount - 1)) + ".png", true, i);
	}
}

void Dungeon::GenerateDungeonLayout()
{
	for (int i = 0; i < MAP_WIDTH * MAP_HEIGHT; i++)
	{
		genLayout[i] = DOOR_NONE;
	}

	genLayout[12] = DOOR4LRTB;

	// D1R, D1B, D2RB
	std::vector<DOOR_TYPES> rooms;

	std::vector<int> room_nums = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };

	for (int k = 0; k < MAP_WIDTH * MAP_HEIGHT; k++)
	{
		int roomGen = room_nums[random(0, room_nums.size() - 1)];
		if (genLayout[roomGen] != DOOR_NONE)
		{
			if (std::find(room_nums.begin(), room_nums.end(), roomGen) != room_nums.end())
				room_nums.erase(std::find(room_nums.begin(), room_nums.end(), roomGen));
			continue;
		}

		rooms.clear();

		int i = 0;
		int j = 0;

		for (int l = 0; l < roomGen; l++)
		{
			j++;
			if (j == MAP_WIDTH)
			{
				j = 0;
				i++;
			}
		}

		bool left = false, right = false, top = false, bottom = false;

		DOOR_TYPES L = DOOR_OUT_OF_BOUNDS, R = DOOR_OUT_OF_BOUNDS, T = DOOR_OUT_OF_BOUNDS, B = DOOR_OUT_OF_BOUNDS;
		if (j - 1 >= 0)
		{
			R = genLayout[i * MAP_WIDTH + j - 1];
			if (DOOR_TYPE_STRINGS[R].find("R") != std::string::npos)
			{
				right = true;
			}
		}
		if (j + 1 <= MAP_WIDTH - 1)
		{
			L = genLayout[i * MAP_WIDTH + j + 1];
			if (DOOR_TYPE_STRINGS[L].find("L") != std::string::npos)
			{
				left = true;
			}
		}
		if (i - 1 >= 0)
		{
			B = genLayout[(i - 1) * MAP_WIDTH + j];
			if (DOOR_TYPE_STRINGS[B].find("B") != std::string::npos)
			{
				bottom = true;
			}
		}
		if (i + 1 <= MAP_HEIGHT - 1)
		{
			T = genLayout[(i + 1) * MAP_WIDTH + j];
			if (DOOR_TYPE_STRINGS[T].find("T") != std::string::npos)
			{
				top = true;
			}
		}

		if ((left && right && top && bottom) ||
			(L == DOOR_NONE && right && top && bottom) ||
			(left && R == DOOR_NONE && top && bottom) ||
			(left && right && T == DOOR_NONE && bottom) ||
			(left && right && top && B == DOOR_NONE) ||
			(left && right && T == DOOR_NONE && B == DOOR_NONE) ||
			(left && top && R == DOOR_NONE && B == DOOR_NONE) ||
			(left && bottom && R == DOOR_NONE && T == DOOR_NONE) ||
			(right && top && L == DOOR_NONE && B == DOOR_NONE) ||
			(right && bottom && L == DOOR_NONE && T == DOOR_NONE) ||
			(top && bottom && L == DOOR_NONE && R == DOOR_NONE) ||
			(left && R == DOOR_NONE && T == DOOR_NONE && B == DOOR_NONE) ||
			(L == DOOR_NONE && right && T == DOOR_NONE && B == DOOR_NONE) ||
			(L == DOOR_NONE && R == DOOR_NONE && top && B == DOOR_NONE) ||
			(L == DOOR_NONE && R == DOOR_NONE && T == DOOR_NONE && bottom) ||
			(L == DOOR_NONE && R == DOOR_NONE && T == DOOR_NONE && B == DOOR_NONE))
			rooms.push_back(DOOR4LRTB); // If all sides have access doors, or > 1 side does and other sides are empty

		if (!bottom)
		{
			if ((left && right && top) ||
				(left && right && T == DOOR_NONE) ||
				(left && top && R == DOOR_NONE) ||
				(top && right && L == DOOR_NONE) ||
				(left && R == DOOR_NONE && T == DOOR_NONE) ||
				(right && L == DOOR_NONE && T == DOOR_NONE) ||
				(top && L == DOOR_NONE && R == DOOR_NONE) ||
				(L == DOOR_NONE && R == DOOR_NONE && T == DOOR_NONE))
				rooms.push_back(DOOR3_LRB);
		}

		if (!top)
		{
			if ((left && right && bottom) ||
				(left && right && B == DOOR_NONE) ||
				(left && bottom && R == DOOR_NONE) ||
				(bottom && right && L == DOOR_NONE) ||
				(left && R == DOOR_NONE && B == DOOR_NONE) ||
				(right && L == DOOR_NONE && B == DOOR_NONE) ||
				(bottom && L == DOOR_NONE && R == DOOR_NONE) ||
				(L == DOOR_NONE && R == DOOR_NONE && B == DOOR_NONE))
				rooms.push_back(DOOR3_LRT);
		}

		if (!right)
		{
			if ((left && top && bottom) ||
				(left && top && B == DOOR_NONE) ||
				(left && bottom && T == DOOR_NONE) ||
				(bottom && top && L == DOOR_NONE) ||
				(left && T == DOOR_NONE && B == DOOR_NONE) ||
				(top && L == DOOR_NONE && B == DOOR_NONE) ||
				(bottom && L == DOOR_NONE && T == DOOR_NONE) ||
				(L == DOOR_NONE && T == DOOR_NONE && B == DOOR_NONE))
				rooms.push_back(DOOR3_RTB);
		}

		if (!left)
		{
			if ((right && top && bottom) ||
				(right && top && B == DOOR_NONE) ||
				(right && bottom && T == DOOR_NONE) ||
				(bottom && top && R == DOOR_NONE) ||
				(right && T == DOOR_NONE && B == DOOR_NONE) ||
				(top && R == DOOR_NONE && B == DOOR_NONE) ||
				(bottom && R == DOOR_NONE && T == DOOR_NONE) ||
				(R == DOOR_NONE && T == DOOR_NONE && B == DOOR_NONE))
				rooms.push_back(DOOR3_LTB);
		}

		if (!top && !bottom)
		{
			if ((left && right) ||
				(left && R == DOOR_NONE) ||
				(right && L == DOOR_NONE) ||
				(L == DOOR_NONE && R == DOOR_NONE))
				rooms.push_back(DOOR2_LR);
		}

		if (!left && !right)
		{
			if ((top && bottom) ||
				(top && B == DOOR_NONE) ||
				(bottom && T == DOOR_NONE) ||
				(T == DOOR_NONE && B == DOOR_NONE))
				rooms.push_back(DOOR2_TB);
		}

		if (!right && !bottom)
		{
			if ((left && top) ||
				(left && T == DOOR_NONE) ||
				(top && L == DOOR_NONE) ||
				(L == DOOR_NONE && T == DOOR_NONE))
				rooms.push_back(DOOR2_RB);
		}

		if (!right && !top)
		{
			if ((left && bottom) ||
				(left && B == DOOR_NONE) ||
				(bottom && L == DOOR_NONE) ||
				(L == DOOR_NONE && B == DOOR_NONE))
				rooms.push_back(DOOR2_RT);
		}

		if (!left && !bottom)
		{
			if ((right && top) ||
				(right && T == DOOR_NONE) ||
				(top && R == DOOR_NONE) ||
				(R == DOOR_NONE && T == DOOR_NONE))
			{
				std::cout << right << " " << top << " " << R << " " << T << std::endl;
				rooms.push_back(DOOR2_LB);
			}
		}

		if (!left && !top)
		{
			if ((right && bottom) ||
				(right && B == DOOR_NONE) ||
				(bottom && R == DOOR_NONE) ||
				(R == DOOR_NONE && B == DOOR_NONE))
				rooms.push_back(DOOR2_LT);
		}

		if ((left && R == DOOR_NONE && T == DOOR_NONE && B == DOOR_NONE) ||
			(left && !right && !top && !bottom))
			rooms.push_back(DOOR1_R);
		if ((L == DOOR_NONE && right && T == DOOR_NONE && B == DOOR_NONE) ||
			(right && !left && !top && !bottom))
			rooms.push_back(DOOR1_L);
		if ((L == DOOR_NONE && R == DOOR_NONE && top && B == DOOR_NONE) ||
			(top && !left && !right && !bottom))
			rooms.push_back(DOOR1_B);
		if ((L == DOOR_NONE && R == DOOR_NONE && T == DOOR_NONE && bottom) ||
			(bottom && !left && !right && !top))
			rooms.push_back(DOOR1_T);

		if (rooms.empty())
			genLayout[i * MAP_WIDTH + j] == DOOR_NONE;
		else
			genLayout[i * MAP_WIDTH + j] = rooms[random(0, rooms.size() - 1)];

		if (std::find(room_nums.begin(), room_nums.end(), roomGen) != room_nums.end())
			room_nums.erase(std::find(room_nums.begin(), room_nums.end(), roomGen));
	}

	std::cout << "BREAK!" << std::endl;
}

void Dungeon::render(sf::RenderWindow& window)
{
	for (auto& room : dungeon)
	{
		room.render(window);
	}
}