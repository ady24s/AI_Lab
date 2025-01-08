a_list = []
a_list.append(11)
a_list.append(20)
a_list.append(155)
a_list.append(78)
print("List after adding elements: ", a_list)

a_list.insert(1,5)
print("List after insertion at index 1: ", a_list)

a_list.remove(155)
print("List after removing element: ", a_list)

a_list.sort()
print("List after sorting: ", a_list)

a_list.reverse()
print("List after reversing: ", a_list)

search_element = input("Enter elemenent to search")
if search_element in a_list:
    print(f"Element {search_element} found at index {a_list.index(search_element)}")
else:
    print(f"Element {search_element} not found")
