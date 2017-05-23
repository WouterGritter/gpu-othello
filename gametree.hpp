#ifndef __GAMETREE_HPP__
#define __GAMETREE_HPP__

#include <vector>
#include "board.hpp"
#include "common.hpp"

// Cp is a constant used for calculating a node's UCT score
#define CP 0.7

class Node {
  public:
    Node(Board b, Node *parent, Side side);
    ~Node();

    // Create Node for the given Child index (same index as move).
    Node *addChild(int i);

    // Search all descendants of this node for the best node to expand
    // based on UCT score. The max score will be returned, and the
    // node itself will be returned using the output variable expandNode.
    Node *searchScore();

    void updateSim(int numSims, int numWins);

    // Given simulations so far, return move with most number of simulations.
    Move getBestMove();

  private:
    bool terminal;
    bool fullyExpanded;
    Side side;
    Board board;
    Node *parent;
    std::vector<Move> moves;
    std::vector<Node*> children;
    int numWins;
    int numSims;
};

#endif