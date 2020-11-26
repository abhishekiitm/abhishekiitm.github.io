title: Count twos
date: 2020-11-26
author: Abhishek Saini
tags: algorithm, counting, combinatorics
category: algorithm
summary: Short version for index and feeds

# Count 2's between 0 and n

Let's look at a very rudimentary algorithm for finding the number of 2's between 0 and n.

## Brute force solution
- We loop through every integer x between zero and n.  
- For each such integer x count the number of 2's in x
- Keep a running sum of the counts to get the final result

Here's the Python code for the same:

```python
def calc_twos_brute(n):
    cnt = 0
    for i in range(n+1):
        l = [int(x) for x in str(i)]
        cnt=cnt+l.count(2)
    return cnt
```

The run time of this algorithm increases proportionally to n (O(n)) and if you test this for a big enough number you will realize how slow this gets. 

However, a brute force solution can be super useful when trying to debug a better solution (which we will look at in a future post).