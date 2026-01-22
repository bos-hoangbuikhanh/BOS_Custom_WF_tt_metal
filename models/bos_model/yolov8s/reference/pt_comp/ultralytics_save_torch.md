

### How to save ultralytics models output

- location: {venv location}/lib/python3.12/site-packages/ultralytics/nn/tasks.py line 149
```
 96:  class BaseModel(torch.nn.Module):
135:      def _predict_once(self, x, profile=False, visualize=False, embed=None):
149:          for m in self.model:
154:              x = m(x)  # run
```

add before run
```
if m.i == 0:
    torch.save(x, "pt_comp/ultra_input.0.pt")
```

add after run
```
torch.save(x, f"pt_comp/ultra_output.{m.i}.pt")
```
