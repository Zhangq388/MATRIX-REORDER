#ifndef __INTEGER_HEAP__
#define __INTEGER_HEAP__
//优先级队列，整数堆
struct Node
{
    int     ID = -1;
    int     batch_id = -1;
    bool    STS =true;
    int     key1 = 0;
    int     key2 = 0;
    Node*   left;
    Node*   right;
    Node() { left = right = this; };
    Node(int _id): ID(_id) { left = right = this; };
    Node(int _id, int _batch_id): ID(_id), batch_id(_batch_id) { left = right = this; };
    bool operator<(const Node& _rhs) const
    {
        return this->key1 + this->key2 < _rhs.key1 + _rhs.key2;
    };
    ~Node() = default;
};

struct PtrNodeLess
{
    bool operator()(const Node* _lhs, const Node* _rhs) const
    {
        return *_lhs < *_rhs;
    };
};

class Integer_Heap
{
private:
    void _remove_(Node*& node)
    {
      Node*&  tmp = HTAB[node->key1];

      if(tmp == node)
      {
          if(node->right == node)
          {
              tmp = nullptr;
          }
          else
          {
              tmp = node->right; 
              node->left->right = node->right;
              node->right->left = node->left;
              node->left = node;
              node->right = node;
          }
      }
      else
      {
          node->left->right = node->right;
          node->right->left = node->left;
          node->left = node;
          node->right = node;
      }
    };

    void _insert_(Node*& node)
    {
      node->key1 += node->key2;
      node->key2 = 0;

      Node*& tmp = HTAB[node->key1];

      if(tmp == nullptr)
      {
          tmp = node;
      }
      else
      {
          node->left = tmp;
          node->right = tmp->right;
          tmp->right->left = node;
          tmp->right = node;
      }
    };
public:
    Integer_Heap(): HTAB(nullptr), PMAX(0), CNT(0) {};
    Integer_Heap(const int n){ HTAB = (Node**)calloc(n, sizeof(Node*)); PMAX = 0; CNT=0;};
    ~Integer_Heap(){free(HTAB); PMAX = 0; CNT=0;};
    void push(Node*& node)
    {
      _insert_(node);
      ++CNT;
      if(node->key1 > PMAX)
      {
          PMAX = node->key1;
      }
    };

    void modify(Node*& node, int val)
    {
      node->key2 += val;
      if(node->key2 > 0)
      {
          _remove_(node);
          _insert_(node);
          if(node->key1 > PMAX)
          {
              PMAX = node->key1;
          }
      }
    };
    
    void refresh(Node*& node)
    {
        if (node->key1 > 0)
        {
            _remove_(node);
            node->key1 = 0;
            _insert_(node);
        }
    };

    void pop(Node*& node)
    {
      Node*   tmp;
      while(CNT>0)
      {
          while(PMAX >0 && HTAB[PMAX] == nullptr)
          {
              --PMAX;
          }
          tmp = HTAB[PMAX];

          while(tmp != nullptr)
          {
              if(tmp->key2 == 0)
              {
                  _remove_(tmp);
                  //tmp->key1 = 0;
                  node = tmp;
                  --CNT;
                  return;
              }
              else
              {
                  _remove_(tmp);
                  _insert_(tmp);
              }

              tmp = HTAB[PMAX];
          }
      }
    };

    Node**                HTAB;
    int                   PMAX;
    int                   CNT;
};
#endif