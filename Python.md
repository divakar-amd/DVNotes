# Python notes, tricks and concepts


# Contents
 - [Finding class object details](#Finding-class-object-details) 
 - 

<br>

### Finding class object details
- ```python
  my_obj = <_your_class_obj_>
  print(type(my_obj))
  print(my_obj.__class__)
  print(my_obj.__class__.__name__)
  print(my_obj.__class__.__module__)
  ```
- More details:
  ```python
  import inspect
  print(f"Class: {my_obj.__class__}")
  print(f"Module: {inspect.getmodule(my_obj)}")
  print(f"Source file: {inspect.getfile(my_obj.__class__)}")
  print(f"MRO: {inspect.getmro(my_obj.__class__)}")
  ```
