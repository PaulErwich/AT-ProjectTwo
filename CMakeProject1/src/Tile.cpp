//
// Created by MLG Erwich on 26/10/2022.
//

#include "Tile.h"
Tile::Tile(sf::Texture& _texture, int x, int y, int texture_num, int _cost, int scale)
{
    tile_coords = std::make_tuple(x, y);

    cost = _cost;

    texture = std::make_unique<sf::Texture>(_texture);

    if (texture_num != -1)
    {
        int left = 0, top = 0;

        for (int l = 0; l < texture_num; l++)
        {
            left++;
            if (left == 5)
            {
                left = 0;
                top++;
            }
        }

        sf::IntRect texture_rect(left * 8, top * 8,
            8, 8);

        sprite->setTexture(*texture);
        sprite->setTextureRect(texture_rect);
        sprite->setScale(scale, scale);
        sprite->setPosition(x * 8 * scale, y * 8 * scale);

        width = sprite->getGlobalBounds().width;
        height = sprite->getGlobalBounds().height;

        setMin();
        calculateMax();
    }
}

std::tuple<int, int> Tile::getCoords() { return tile_coords; }
int Tile::getCost() const { return cost; }
