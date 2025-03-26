#pragma once

#include <SFML/Graphics.hpp>
#include <iostream>

class GameObject
{
public:
	GameObject();
	~GameObject() = default;

	virtual bool init();

	sf::Sprite& getSprite();

	virtual void update(float dt);
	void render(sf::RenderWindow& window);

	void setMin();
	sf::Vector2f getMin();

	void calculateMax();
	sf::Vector2f getMax();

	float getWidth() const;
	float getHeight() const;

	bool getVisible() const;
	void setVisible(bool vis);

	sf::Vector2f getMovement();
	void setMovement(sf::Vector2f move);

	bool AABBCollision(GameObject& collider) const;

protected:
	std::unique_ptr<sf::Sprite> sprite;
	std::unique_ptr<sf::Texture> texture;

	sf::Vector2f movement;

	sf::Vector2f min;
	sf::Vector2f max;
	float width;
	float height;

	bool visible;
};
