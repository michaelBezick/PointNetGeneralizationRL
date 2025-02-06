# Variables
REMOTE = origin          # Change to your remote name (default is 'origin')
BRANCH = main            # Change to your branch name (default is 'main')
MESSAGE = "Auto commit"  # Default commit message

# Rules
all: push

add:
	git add --all

commit:
	git commit -m $(MESSAGE)

push: add commit
	git push $(REMOTE) $(BRANCH)

# Rule to specify a custom commit message
push-with-message:
	git add --all
	git commit -m "$(MESSAGE)"
	git push $(REMOTE) $(BRANCH)

# Rule to specify commit message as a parameter
push-message:
	@echo "Enter commit message: "; \
        read msg; \
	git add --all; \
	git commit -m "$$msg"; \
	git push $(REMOTE) $(BRANCH)
