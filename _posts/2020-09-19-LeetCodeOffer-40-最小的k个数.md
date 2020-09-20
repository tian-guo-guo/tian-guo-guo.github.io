---
layout:     post           # 使用的布局（不需要改）
title:      LeetCode-Offer40-最小的k个数
subtitle:   LeetCode-Offer40-最小的k个数 #副标题
date:       2020-09-19            # 时间
author:     甜果果                    # 作者
header-img: https://cdn.jsdelivr.net/gh/tian-guo-guo/cdn@master/assets/picgoimg/20200701171155.png  #背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - LeetCode
    - python
    - Offer

---

# [剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

tag: easy，数组

**题目：**

输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

 **示例一**：

```
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
```

 **示例二：**

```
输入：arr = [0,1,2,1], k = 1
输出：[0]
```



- 基础排序算法总结
    - 交换类：冒泡排序、快速排序
    - 选择类：简单选择排序、快速排序
    - 插入类：直接插入排序、shell 排序
    - 归并类：归并排序

# 方法一：交换类排序 -- 冒泡排序

-   思想：从无序序列头部开始，进行两两比较，根据大小交换位置，直到最后将最大（小）的数据元素交换到了无序队列的队尾，从而成为有序序列的一部分；下一次继续这个过程，直到所有数据元素都排好序。

-   时间复杂度：O(n2),空间复杂度 O(1)
-   稳定性： 稳定

```python
class Solution1:
    @staticmethod
    def swap(nums, i, j):
        nums[i], nums[j] = nums[j], nums[i]

    def bubble_sort(self, nums):
        for i in range(1, len(nums)):  # 控制比较趟数，最多比较 len(nums) - 1趟
            flag = False  # 比较当前趟是否发生交换，没有则已经排序好了
            for j in range(0, len(nums) - i):
                if nums[j] > nums[j + 1]:
                    flag = True
                    self.swap(nums, j, j + 1)
            if not flag: return nums

        return nums

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0: return []
        if len(arr) <= k: return arr

        return self.bubble_sort(arr)[:k]
```



# 方法二：交换类排序 -- 快速排序

- 思想：分别从初始序列“**6 1 2 7 9 3 4 5 10 8**”两端开始“探测”。先从**右**往**左**找一个小于 **6** 的数，再从**左**往**右**找一个大于 **6** 的数，然后交换他们。这里可以用两个变量 **i** 和 **j**，分别指向序列最左边和最右边。我们为这两个变量起个好听的名字“哨兵 i”和“哨兵 j”。刚开始的时候让哨兵 i 指向序列的最左边（即 **i=1**），指向数字 **6**。让哨兵 **j** 指向序列的最右边（即 **j=10**），指向数字 **8**。

    首先哨兵 **j** 开始出动。因为此处设置的基准数是最左边的数，所以需要让哨兵 **j** 先出动，这一点非常重要（请自己想一想为什么）。哨兵 **j** 一步一步地向左挪动（即 **j--**），直到找到一个小于 **6** 的数停下来。接下来哨兵 **i** 再一步一步向右挪动（即 **i++**），直到找到一个数大于 **6** 的数停下来。最后哨兵 **j** 停在了数字 **5** 面前，哨兵 **i** 停在了数字 **7** 面前。

    现在交换哨兵 **i** 和哨兵 **j** 所指向的元素的值。交换之后的序列如下。6 1 2 **5** 9 3 4 **7** 10 8

    到此，第一次交换结束。接下来开始哨兵 **j** 继续向左挪动（再友情提醒，每次必须是哨兵 **j** 先出发）。他发现了 **4**（比基准数 **6** 要小，满足要求）之后停了下来。哨兵 **i** 也继续向右挪动的，他发现了 **9**（比基准数 **6** 要大，满足要求）之后停了下来。此时再次进行交换，交换之后的序列如下。 6 1 2 5 **4** 3 **9** 7 10 8

    第二次交换结束，“探测”继续。哨兵 **j** 继续向左挪动，他发现了 **3**（比基准数 **6** 要小，满足要求）之后又停了下来。哨兵 **i** 继续向右移动，糟啦！此时哨兵 **i** 和哨兵 **j** 相遇了，哨兵 **i** 和哨兵 **j** 都走到 **3** 面前。说明此时“探测”结束。我们将基准数 **6** 和 **3** 进行交换。交换之后的序列如下。 **3** 1 2 5 4 **6** 9 7 10 8

    每次排序的时候设置一个基准点，将小于等于基准点的数全部放到基准点的左边，将大于等于基准点的数全部放到基准点的右边。

-   时间复杂度：O(nlogn),空间复杂度 O(nlogn)
-   稳定性： 不稳定

```python
class Solution2:
    @staticmethod
    def swap(nums, i, j):
        nums[i], nums[j] = nums[j], nums[i]

    def partition(self, nums, left, right):
        pivot = left
        i = j = pivot + 1

        while j <= right:
            if nums[j] <= nums[pivot]:
                self.swap(nums, i, j)
                i += 1
            j += 1
        self.swap(nums, pivot, i - 1)
        return i - 1

    def quick_sort(self, nums, left, right):
        if left < right:
            pivot = self.partition(nums, left, right)
            self.quick_sort(nums, left, pivot - 1)
            self.quick_sort(nums, pivot + 1, right)

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0: return []
        if len(arr) <= k: return arr

        self.quick_sort(arr, 0, len(arr) - 1)
        return arr[:k]
```



# 方法三：选择类排序 -- 简单选择排序

-   思想：每一趟从待排序的数据元素中选择最小（或最大）的一个元素作为首元素，直到所有元素排完为止
-   时间复杂度：O(n2),空间复杂度 O(1)
-   稳定性： 不稳定

```python
class Solution3:
    @staticmethod
    def swap(nums, i, j):
        nums[i], nums[j] = nums[j], nums[i]

    # 循环迭代地将后面未排序序列最小值交换到前面有序序列末尾
    def select_sort(self, nums):
        for i in range(len(nums) - 1):
            min_index = i
            for j in range(i + 1, len(nums)):
                if nums[j] < nums[min_index]:
                    min_index = j
            if min_index != i:
                self.swap(nums, i, min_index)
        return nums

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0: return []
        if len(arr) <= k: return arr

        return self.select_sort(arr)[:k]
```



# 方法四：选择类排序 -- 堆排序

-   [思想](https://www.cnblogs.com/chengxiao/p/6129630.html)：**将待排序序列构造成一个大顶堆，此时，整个序列的最大值就是堆顶的根节点。将其与末尾元素进行交换，此时末尾就为最大值。然后将剩余n-1个元素重新构造成一个堆，这样会得到n个元素的次小值。如此反复执行，便能得到一个有序序列了**
-   时间复杂度：O(nlogn),空间复杂度 O(1)
-   稳定性： 不稳定

```python
class Solution4:
    @staticmethod
    def swap(nums, i, j):
        nums[i], nums[j] = nums[j], nums[i]

    def heapify(self, nums, i, size):
        left, right = 2 * i + 1, 2 * i + 2

        largest = i
        if left < size and nums[left] > nums[largest]:
            largest = left
        if right < size and nums[right] > nums[largest]:
            largest = right

        if largest != i:
            self.swap(nums, i, largest)
            self.heapify(nums, largest, size)

    def build_heap(self, nums, k):
        # 从最后一个非叶子节点开始堆化
        for i in range(k // 2 - 1, -1, -1):
            self.heapify(nums, i, k)

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0: return []
        if len(arr) <= k: return arr

        heap = arr[:k]
        self.build_heap(heap, k)

        for i in range(k, len(arr)):
            if arr[i] < heap[0]:
                heap[0] = arr[i]
                self.heapify(heap, 0, k)
        return heap
```



# 方法五：插入类排序 -- 直接插入排序

-   思想：把n个待排序的元素看成为一个有序表和一个无序表。开始时有序表中只包含1个元素，无序表中包含有n-1个元素，排序过程中每次从无序表中取出第一个元素，将它插入到有序表中的适当位置，使之成为新的有序表，重复n-1次可完成排序过程。
-   时间复杂度：O(n2),空间复杂度 O(1)
-   稳定性： 不稳定

```python
class Solution5:
    def insert_sort(self, nums):
        for i in range(1, len(nums)):
            cur_value = nums[i]
            pos = i
            while pos >= 1 and nums[pos - 1] > cur_value:
                nums[pos] = nums[pos - 1]
                pos -= 1
            nums[pos] = cur_value
        return nums

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0: return []
        if len(arr) <= k: return arr

        return self.insert_sort(arr)[:k]
```



# 方法六：插入类排序 -- shell 排序

-   [思想](https://www.cnblogs.com/chengxiao/p/6104371.html)：**希尔排序是把记录按下标的一定增量分组，对每组使用直接插入排序算法排序；随着增量逐渐减少，每组包含的关键词越来越多，当增量减至1时，整个文件恰被分成一组，算法便终止。**
-   时间复杂度：O(n1.3),空间复杂度 O(1)
-   稳定性： 不稳定

```python
class Solution6:
    # 分组插入排序
    def insert_sort_gap(self, nums, start, gap):
        for i in range(start + gap, len(nums), gap):
            cur_value = nums[i]
            pos = i
            while pos >= gap and nums[pos - gap] > cur_value:
                nums[pos] = nums[pos - gap]
                pos -= gap
            nums[pos] = cur_value

    def shell_sort(self, nums):
        gap = len(nums) // 2
        while gap > 0:
            for i in range(gap):
                self.insert_sort_gap(nums, i, gap)
            gap //= 2
        return nums

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0: return []
        if len(arr) <= k: return arr

        return self.shell_sort(nums)[:k]
```



# 方法七：归并类排序 -- 归并排序

-   [思想](https://www.cnblogs.com/chengxiao/p/6194356.html)：归并排序（MERGE-SORT）是利用**归并**的思想实现的排序方法，该算法采用经典的**分治**（divide-and-conquer）策略（分治法将问题**分**(divide)成一些小的问题然后递归求解，而**治(conquer)**的阶段则将分的阶段得到的各答案"修补"在一起，即分而治之)。
-   时间复杂度：O(nlogn),空间复杂度 O(1)
-   稳定性： 不稳定

```python
class Solution7:
    def merge(self, l1, l2):
        res = []
        i, j = 0, 0
        while i < len(l1) and j < len(l2):
            if l1[i] <= l2[j]:
                res.append(l1[i])
                i += 1
            else:
                res.append(l2[j])
                j += 1
        if i == len(l1):
            res.extend(l2[j:])
        else:
            res.extend(l1[i:])
        return res

    def merge_sort(self, nums):
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2
        left = self.merge_sort(nums[:mid])
        right = self.merge_sort(nums[mid:])
        return self.merge(left, right)

    def getLeastNumbers(self, arr, k: int):
        if not arr or k <= 0: return []
        if len(arr) <= k: return arr

        return self.merge_sort(arr)[:k]
```

