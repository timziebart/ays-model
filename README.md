## argument completion

aws-model uses argcomplete, so if you want to have argument completion please run once

    activate-global-python-argcomplete --dest=- >> ~/.bash_completion

afterwards you should be able to use [Tab] for autocompletion

## security note

this code is prepared for scientific work, not a production where security and tight right management ist important. The `eval` function is used for simplicity during command line evaluation ...
