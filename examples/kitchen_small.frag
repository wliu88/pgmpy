// Priors
IsA(x,Bowl.n.03)
IsA(x,Fork.n.01)

// Posteriors
IsA(x,Bowl.n.03) -> IsA(x,Container.n.01)
IsA(x,Container.n.01) -> AtLocation(x,Garage)
IsA(x,Bowl.n.03) -> AtLocation(x,Sink)
IsA(x,Tableware.n.01) -> IsA(x,Article.n.02)
IsA(x,Fork.n.01) ~ IsA(x,Bowl.n.03) -> IsA(x,Tableware.n.01)
IsA(x,Fork.n.01) -> AtLocation(x,Bathroom)
IsA(x,Fork.n.01) ~ IsA(x,Bowl.n.03) -> AtLocation(x,Table)
