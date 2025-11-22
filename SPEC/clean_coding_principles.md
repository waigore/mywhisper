# Clean Coding Principles

## Purpose

This document defines the clean coding principles that all code in the mywhisper project must adhere to. These principles are enforced through code review and should be referenced in all module specifications.

## Principles

### 1. Build Abstractions and Implement Delegation + Encapsulation

**Principle:** Always build abstractions and implement delegation and encapsulation whenever possible. High-level components should never need to know implementation details of lower-level components.

**Examples:**
- The pipeline should never have to execute individual procedures for each step. Instead, steps should be abstracted behind a common interface that the pipeline can delegate to.
- Components should encapsulate their internal state and expose only necessary interfaces.
- Use factory patterns, dependency injection, and interface abstractions to decouple components.

**Enforcement:**
- When reviewing code, verify that high-level orchestration code (like `PipelineRunner`) does not contain step-specific logic.
- Check that each step implements a common interface/contract rather than requiring custom handling in the pipeline.
- Ensure that internal implementation details are not exposed beyond module boundaries.

### 2. Avoid Nested Conditionals - Use Conditional Guards

**Principle:** Nested conditionals are a code smell and should be avoided. Prefer conditional guards (early returns/continues) to flatten control logic. If control logic is still too complicated (e.g., 3 or more nested layers), abstract it away into a class or function.

**Rule:** Never have more than 2 levels of nested if statements.

**Examples:**

**Bad:**
```python
if condition_a:
    if condition_b:
        if condition_c:
            # do work
        else:
            pass
    else:
        pass
else:
    pass
```

**Better (with guards):**
```python
if not condition_a:
    return  # or continue, or handle early case

if not condition_b:
    return

if not condition_c:
    return

# do work
```

**When to Abstract:**
- If after applying guard clauses, you still have 3+ levels of nesting, extract the logic into a separate function or class.
- Complex conditional logic that represents a distinct decision point should be encapsulated in a method with a descriptive name.

**Enforcement:**
- Code reviews must flag any code with more than 2 levels of nested conditionals.
- Prefer guard clauses (`if not condition: return/continue`) over nested if-else blocks.
- Extract complex conditional logic into well-named helper methods or classes.

### 3. Avoid Inline Imports

**Principle:** Avoid inline imports (imports inside functions or methods) as they build hidden dependencies and contribute to undesired coupling. All imports should be at the module level.

**Rationale:**
- Inline imports make dependencies less visible and harder to track.
- They can create circular import issues.
- They make it difficult to understand module dependencies at a glance.
- They can lead to performance issues if imports happen in hot paths.

**Examples:**

**Bad:**
```python
def process_episode(episode):
    from mywhisper.transcribe import PodcastTranscriber
    transcriber = PodcastTranscriber.from_config(config)
    # ...
```

**Better:**
```python
from mywhisper.transcribe import PodcastTranscriber

def process_episode(episode):
    transcriber = PodcastTranscriber.from_config(config)
    # ...
```

**Exceptions:**
- Only acceptable when importing inside a function is necessary to break a circular import, and this should be documented with a comment explaining why.

**Enforcement:**
- All imports must be at the top of the file (after module docstring and before any other code).
- Code reviews must flag any inline imports and require justification if they cannot be moved to module level.

### 4. Avoid hasattr Unless Absolutely Necessary

**Principle:** Avoid using `hasattr()` unless absolutely necessary. `hasattr` degrades code quality and makes the program more brittle during execution.

**Rationale:**
- `hasattr` relies on dynamic attribute access, which bypasses static type checking and makes code harder to reason about.
- It creates implicit contracts that are not enforced by the type system or class definitions.
- Code using `hasattr` is more prone to runtime errors that could be caught earlier with proper type checking or explicit interfaces.
- It encourages a "duck typing gone wrong" approach where objects are expected to have certain attributes without clear guarantees.
- Makes refactoring more dangerous as attribute existence checks can silently fail or pass incorrectly.

**Examples:**

**Bad:**
```python
def process_step(step):
    if hasattr(step, 'validate'):
        step.validate()
    if hasattr(step, 'preprocess'):
        step.preprocess()
    step.execute()
```

**Better (using interfaces/abstract base classes):**
```python
from abc import ABC, abstractmethod

class Step(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    def validate(self):
        """Optional validation, can be overridden."""
        pass
    
    def preprocess(self):
        """Optional preprocessing, can be overridden."""
        pass

def process_step(step: Step):
    step.validate()
    step.preprocess()
    step.execute()
```

**Better (using explicit type checking):**
```python
from typing import Protocol

class ValidatableStep(Protocol):
    def validate(self) -> None: ...
    def execute(self) -> None: ...

def process_step(step: ValidatableStep):
    step.validate()
    step.execute()
```

**When hasattr is Acceptable:**
- Only when dealing with truly dynamic objects where attribute existence cannot be determined statically (e.g., parsing external data structures, working with third-party libraries that don't provide type hints).
- When the alternative would require significant architectural changes that are not feasible in the short term (should be documented and marked for refactoring).

**Enforcement:**
- Code reviews must flag all uses of `hasattr` and require justification.
- Prefer abstract base classes, protocols, or explicit type checking to define expected interfaces.
- Use optional methods with default implementations in base classes rather than checking for attribute existence.
- Document any legitimate uses of `hasattr` with comments explaining why it's necessary.

## Integration with Specifications

All module specifications should reference this document and explicitly state how the module adheres to these principles:

- **Abstraction/Delegation:** Describe the interfaces and abstractions the module provides.
- **Control Flow:** Note any complex conditional logic and how it's been flattened or abstracted.
- **Dependencies:** List all module-level imports and explain the module's dependencies.
- **Type Safety:** Document any use of `hasattr` and justify why it's necessary, or describe the explicit interfaces/protocols used instead.

## Code Review Checklist

When reviewing code, verify:

- [ ] No more than 2 levels of nested conditionals
- [ ] Guard clauses are used to flatten control flow
- [ ] Complex logic is abstracted into functions/classes
- [ ] All imports are at module level (no inline imports)
- [ ] High-level components delegate to abstractions rather than implementing details
- [ ] Dependencies are explicit and visible
- [ ] `hasattr` is avoided unless absolutely necessary (with documented justification)
- [ ] Interfaces are defined using abstract base classes, protocols, or explicit type checking

## References

- See `SPEC/myw_spec.md` for architecture-level abstractions
- See `SPEC/spec.md` for module-level requirements
- These principles apply to all code in the mywhisper project

