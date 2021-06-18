import Store  # i don't believe this causes problems as it's defined in ToolAnalysis

def Initialise(pyinit):
    print("i've initialized")
    return 1

def Execute():
    print("i'm executing")
    
    print("getting int from BoostStore")
    state = Store.GetStoreVariable('CStore','state')
    print("i got state value of ",state)
    
    print("getting string from BoostStore")
    astring = Store.GetStoreVariable('CStore','astring')
    print("i got a value of ",astring)
    
    print("getting vector of doubles from BoostStore")
    dubs = Store.GetStoreVariable("CStore","dubs")
    print("i got doubles: {",dubs,"}")
    
    print("getting a position object from a BoostStore")
    pos = Store.GetStoreVariable("CStore","pos")
    print("i got a position of ",pos)
    
    print("Getting an int from an ASCII store")
    state2 = Store.GetStoreVariable("vars","state")
    print("i got a value of ",state2)
    
    print("Getting a string from an ASCII store")
    astring2 = Store.GetStoreVariable("vars","astring")
    print("i got a value of ",astring2)
    
    return 1

def Finalise():
    print("i've finalised")
    return 1
