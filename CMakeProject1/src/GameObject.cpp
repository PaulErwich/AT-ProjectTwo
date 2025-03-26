#include "GameObject.h"

GameObject::GameObject() : width(0), height(0), visible(true),
sprite(new sf::Sprite), texture(new sf::Texture)
{

}

bool GameObject::init()
{
    if (!texture->loadFromFile("")) { return false; }

    sprite->setTexture(*texture);

    width = sprite->getGlobalBounds().width;
    height = sprite->getGlobalBounds().height;

    setMin();
    calculateMax();

    return true;
}

sf::Sprite& GameObject::getSprite() { return *sprite; }

void GameObject::update(float dt) {}
void GameObject::render(sf::RenderWindow& window)
{
    if (visible)
    {
        window.draw(*sprite);
    }
}

void GameObject::setMin() { min = sprite->getPosition(); }
sf::Vector2f GameObject::getMin() { return min; }

void GameObject::calculateMax()
{
    max.x = min.x + width;
    max.y = min.y + height;
}
sf::Vector2f GameObject::getMax() { return max; }

float GameObject::getWidth() const { return width; }
float GameObject::getHeight() const { return height; }

bool GameObject::getVisible() const { return visible; }
void GameObject::setVisible(bool vis) { visible = vis; }

sf::Vector2f GameObject::getMovement() { return movement; }
void GameObject::setMovement(sf::Vector2f move) { movement = move; }

bool GameObject::AABBCollision(GameObject& collider) const
{
    if (max.x <= collider.min.x || min.x >= collider.max.x) return false;
    if (max.y <= collider.min.y || min.y >= collider.max.y) return false;

    return true;
}