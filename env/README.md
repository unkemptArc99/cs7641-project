To install Anaconda, you simply have to download a shell script from the [Anaconda Documentation](https://docs.anaconda.com/anaconda/install/linux/), where you will also find the exact steps for installing the environment.

TIP - On step 7, do initialize the Anaconda3 by running `conda init` through the installer script. But then on step 11, diable the `auto_activate_base` config in Anaconda.

After you have installed Anaconda, simply create the environment `ml_project` from the `.yml ' file in the folder by running the following commands -
```
conda env create -f ml_project.yml
```

After creation of the environment, whenever you want to work on BuzzBlogBenchmark, simply run `conda activate ml_project`. After your work is finished, you can deactivate it by `conda deactivate`.

If you wish to remove this environment from your system, then simply run `conda env remove -n ml_project` after deactivativating the environment.
