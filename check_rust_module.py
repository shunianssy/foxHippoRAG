import foxhipporag_rust
print("Module attributes:", dir(foxhipporag_rust))
print("\nHas cosine_similarity:", hasattr(foxhipporag_rust, 'cosine_similarity'))
print("Has top_k_indices:", hasattr(foxhipporag_rust, 'top_k_indices'))
print("Has __version__:", hasattr(foxhipporag_rust, '__version__'))
