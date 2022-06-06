
Issues:

 
-- DOCUMENTER_KEY
```                                    
using Pkg; Pkg.add("DocumenterTools");                                 
using DocumenterTools                                                  
DocumenterTools.genkeys(user="PetrKryslUCSD", repo="git@github.com:PetrKryslUCSD/Sparspak.jl.git")                                                  
using Pkg; Pkg.rm("DocumenterTools");  
```

$ wsl --set-default-version 2
For information on key differences with WSL 2 please visit https://aka.ms/wsl2             
The operation completed successfully.                                                      

pkonl@Hedwig MINGW64 ~/Documents/00WIP/Sparspak.jl (main)                                  
$ wsl -l -v
  NAME            STATE           VERSION                                                  
* Ubuntu-22.04    Stopped         1                                                        
  Ubuntu-18.04    Stopped         1                                                        

pkonl@Hedwig MINGW64 ~/Documents/00WIP/Sparspak.jl (main)                                  
$ wsl --set-version Ubuntu-22.04 2  
