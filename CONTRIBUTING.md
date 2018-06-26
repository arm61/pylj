## Contributing

If you are interested in contributing to pylj. Please feel free; fork the code and go wild to your hearts content. 

#### Notes

- We are trying to use a modular system within pylj, where there is the overarching `util` module that contains the `System` class which holds all system information. Examples of modules that use the `System` class are the `md` and `mc` module that facilite molecular dynamics and Monte-Carlo simulation respectively. 
- All visuallisation should be done in the `sample` module, the structuring of this module should be clear and generally the only variable required should be the `System` class object. 
- If you want any help implementing your idea, please feel free to discuss it on our [gitter channel](https://gitter.im/pylj/Lobby#). We prefer this method of communication over email, however if you are morally or structurally aposed to gitter feel free to email [arm61](mailto:arm61@bath.ac.uk).
- If you would like to offer a pull request, we will try our best to assess and merge them as appropriate in a timely manner. 
