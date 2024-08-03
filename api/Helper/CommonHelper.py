
class CommonHelper:
    def to_dict(obj):
        dict = {}
        for field in obj.__dict__.keys():
            if not field.startswith("_"):  # Exclude private attributes (optional)
                value = getattr(obj, field)
                dict[field] = value
        return dict